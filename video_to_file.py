#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decodificador compatível com FileToVideo2K (encoder fornecido).
Restaura o arquivo embutido no vídeo criado pelo encoder.
"""

import cv2
import numpy as np
import struct
import zlib
import os
from tqdm import tqdm
import argparse
import math

MAGIC = b'2KBITS01'  # mesmo magic do encoder
DEFAULT_BLOCK_SIZE = 4
DEFAULT_SYNC_FRAMES = 2

def bits_to_bytes(bits):
    """Converte lista de bits (MSB primeiro por byte) para bytes."""
    b = bytearray()
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) < 8:
            # padding zeros (não deve acontecer para o header e dados, mas por segurança)
            byte_bits += [0] * (8 - len(byte_bits))
        val = 0
        for bit in byte_bits:
            val = (val << 1) | int(bool(bit))
        b.append(val)
    return bytes(b)

def extract_bits_from_frame(frame, block_size=4, threshold=128):
    """
    Extrai bits de um frame seguindo o mesmo layout do encoder:
    percorre blocos e amostra a região central do bloco para decidir 1/0.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blocks_x = w // block_size
    blocks_y = h // block_size

    bits = []
    border_default = max(1, block_size // 4)
    for by in range(blocks_y):
        y0 = by * block_size
        y1 = y0 + block_size
        for bx in range(blocks_x):
            x0 = bx * block_size
            x1 = x0 + block_size
            block = gray[y0:y1, x0:x1]

            border = border_default
            # região central (remove borda) - quando possível
            if block_size - 2*border > 0:
                center = block[border: block_size-border, border: block_size-border]
            else:
                center = block  # bloco muito pequeno, usa todo bloco

            # usa média da região central (mais robusto a compressão)
            mean_val = float(np.mean(center))
            bit = 1 if mean_val > threshold else 0
            bits.append(bit)

    return bits

def compute_crc_for_header_fields(filename_len_byte, filename_bytes, file_size, total_bits, num_data_frames):
    """
    Recria o crc usado pelo encoder (crc sobre: filename_len + filename + file_size + total_bits + num_data_frames).
    """
    crc_data = struct.pack('<B', filename_len_byte) + filename_bytes
    crc_data += struct.pack('<Q', file_size)
    crc_data += struct.pack('<Q', total_bits)
    crc_data += struct.pack('<I', num_data_frames)
    crc = zlib.crc32(crc_data) & 0xFFFFFFFF
    return crc

def decode_video(input_video, output_dir='.', block_size=DEFAULT_BLOCK_SIZE, sync_frames=DEFAULT_SYNC_FRAMES, threshold=128, verbose=True):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"✗ Não foi possível abrir o vídeo: {input_video}")
        return False

    # 1) Pula frames de sincronização (encoder usa 2)
    for i in range(sync_frames):
        ret, _ = cap.read()
        if not ret:
            print("✗ Vídeo curto demais (faltam frames de sincronização).")
            cap.release()
            return False
    # 2) Lê primeiro frame após sync para detectar resolução e inicializar parâmetros
    ret, first_frame = cap.read()
    if not ret:
        print("✗ Não há frames após sincronização.")
        cap.release()
        return False

    height, width = first_frame.shape[:2]
    blocks_x = width // block_size
    blocks_y = height // block_size
    bits_per_frame = blocks_x * blocks_y

    if verbose:
        print("=== Decodificador 2K ===")
        print(f"Arquivo de entrada: {input_video}")
        print(f"Resolução detectada: {width}x{height}")
        print(f"Block size: {block_size}x{block_size}")
        print(f"Blocks (x,y): {blocks_x} x {blocks_y}")
        print(f"Bits por frame: {bits_per_frame:,}")

    # vamos processar frames: o first_frame é o primeiro frame de cabeçalho
    frames_bits = []
    # processa first_frame e guarda como primeiro frame de cabeçalho
    bits = extract_bits_from_frame(first_frame, block_size=block_size, threshold=threshold)
    if len(bits) != bits_per_frame:
        print("⚠ Aviso: bits extraídos diferente de bits_per_frame (problema de dimensionamento).")
    frames_bits.append(bits)

    # Agora lemos frames até conseguirmos determinar o tamanho completo do cabeçalho (header)
    header_parsed = False
    filename = None
    file_size = None
    total_data_bits = None
    num_data_frames = None
    header_frames_needed = None
    header_total_bytes = None

    # loop para coletar frames de cabeçalho suficientes
    while True:
        # concatena bits dos frames lidos até agora
        concat_bits = []
        for fb in frames_bits:
            concat_bits.extend(fb)
        # precisamos de pelo menos 10 bytes (8 magic + 1 version + 1 filename_len)
        if len(concat_bits) >= 8*10:
            # pega primeiros 10 bytes para ler filename_len
            first_10_bytes = bits_to_bytes(concat_bits[:8*10])
            magic = first_10_bytes[:8]
            if magic != MAGIC:
                # se magic não bater, é provável que o vídeo não seja do encoder ou sync detectado errado
                print("✗ Magic mismatch: este vídeo não parece ter sido codificado pelo encoder esperado.")
                cap.release()
                return False

            version = first_10_bytes[8]
            filename_len = first_10_bytes[9]
            # calcula tamanho completo do cabeçalho em bytes:
            header_total_bytes = 8 + 1 + 1 + filename_len + 8 + 8 + 4 + 4  # conforme encoder
            header_total_bits = header_total_bytes * 8

            # quantos frames já precisamos para conter todo o header (o encoder alinhou a 64 bytes, mas aqui basta alinhar por frames)
            header_frames_needed = math.ceil(header_total_bits / bits_per_frame)

            if verbose:
                print(f"Magic OK, versão {version}, filename_len = {filename_len}")
                print(f"Header total (bytes): {header_total_bytes}, frames de header necessários: {header_frames_needed}")

            # se já lemos frames suficientes para preencher header_frames_needed, paramos
            if len(frames_bits) >= header_frames_needed:
                # extrai exatamente os bits de header (primeiros header_total_bits)
                header_concat_bits = []
                for i in range(header_frames_needed):
                    header_concat_bits.extend(frames_bits[i])
                header_bytes = bits_to_bytes(header_concat_bits[:header_total_bits])

                # parse header
                # layout: magic(8) version(1) filename_len(1) filename(filename_len) file_size(Q) total_bits(Q) num_data_frames(I) crc(I)
                offset = 0
                magic_read = header_bytes[offset:offset+8]; offset += 8
                version = header_bytes[offset]; offset += 1
                filename_len = header_bytes[offset]; offset += 1
                filename_bytes = header_bytes[offset: offset + filename_len]; offset += filename_len
                file_size = struct.unpack_from('<Q', header_bytes, offset)[0]; offset += 8
                total_data_bits = struct.unpack_from('<Q', header_bytes, offset)[0]; offset += 8
                num_data_frames = struct.unpack_from('<I', header_bytes, offset)[0]; offset += 4
                crc_read = struct.unpack_from('<I', header_bytes, offset)[0]; offset += 4

                # valida CRC
                crc_calc = compute_crc_for_header_fields(filename_len, filename_bytes, file_size, total_data_bits, num_data_frames)
                if crc_calc != crc_read:
                    print("⚠ Aviso: CRC do cabeçalho inválido. O cabeçalho pode estar corrompido.")
                    # prossegue mesmo assim (opcional) — aqui vou abortar para evitar sobrescrever algo
                    cap.release()
                    return False

                filename = filename_bytes.decode('utf-8', errors='replace')
                if verbose:
                    print("=== Cabeçalho lido ===")
                    print(f"Arquivo: {filename}")
                    print(f"Tamanho (bytes): {file_size}")
                    print(f"Bits originais: {total_data_bits}")
                    print(f"Frames de dados esperados: {num_data_frames}")

                header_parsed = True
                break  # já temos header completo
        # se ainda não lemos frames suficientes, lemos mais frames
        ret, frame = cap.read()
        if not ret:
            print("✗ Vídeo terminou antes de obter cabeçalho completo.")
            cap.release()
            return False
        frames_bits.append(extract_bits_from_frame(frame, block_size=block_size, threshold=threshold))
        # evitamos loop infinito; o while termina quando header_parsed ou não há frames

    # header_parsed True e header_frames_needed definido
    # já temos frames_bits[0:header_frames_needed] = frames de header
    # agora precisamos ler exatamente num_data_frames frames adicionais para os dados
    data_frames_collected = 0
    # se já lermos frames extras além do header (pode acontecer se capturamos mais enquanto detectamos), contamos-os:
    frames_total = len(frames_bits)
    extra_after_header = frames_total - header_frames_needed
    if extra_after_header > 0:
        # esses frames adicionais contam como parte dos frames de dados
        data_frames_collected = extra_after_header

    # já colocamos os frames extras (se houver) no início dos frames de dados.
    # agora lemos os frames restantes necessários
    if verbose:
        print(f"Lendo {num_data_frames} frames de dados (já coletados {data_frames_collected}).")

    # read remaining frames
    pbar = tqdm(total=num_data_frames, desc="Frames de dados", unit="fr") if verbose else None
    # decrement pbar for frames already collected
    if pbar is not None:
        pbar.update(data_frames_collected)

    while data_frames_collected < num_data_frames:
        ret, frame = cap.read()
        if not ret:
            print("✗ Vídeo terminou antes de obter todos os frames de dados.")
            if pbar: pbar.close()
            cap.release()
            return False
        frames_bits.append(extract_bits_from_frame(frame, block_size=block_size, threshold=threshold))
        data_frames_collected += 1
        if pbar is not None:
            pbar.update(1)
    if pbar:
        pbar.close()

    cap.release()

    # agora concatena todos os frames de dados (frames_bits[header_frames_needed : header_frames_needed + num_data_frames])
    data_concat_bits = []
    for i in range(header_frames_needed, header_frames_needed + num_data_frames):
        data_concat_bits.extend(frames_bits[i])

    # original file bits = primeiros total_data_bits bits
    if len(data_concat_bits) < total_data_bits:
        print("✗ Não há bits de dados suficientes recuperados.")
        return False

    original_bits = data_concat_bits[:total_data_bits]
    # converte bits para bytes
    data_bytes = bits_to_bytes(original_bits)

    # valida tamanho do arquivo
    if len(data_bytes) != file_size:
        # pode ocorrer diferença se houve truncamento; reduz ou reporta
        if len(data_bytes) > file_size:
            data_bytes = data_bytes[:file_size]
            print("⚠ Ajustado tamanho do arquivo recuperado (cortado para file_size).")
        else:
            print(f"⚠ Tamanho recuperado ({len(data_bytes)}) difere do file_size ({file_size}).")

    # salva arquivo
    out_path = os.path.join(output_dir, filename)
    # se arquivo já existe, evita sobrescrever sem aviso - adiciona sufixo
    base, ext = os.path.splitext(out_path)
    final_out = out_path
    idx = 1
    while os.path.exists(final_out):
        final_out = f"{base}_rec{idx}{ext}"
        idx += 1

    with open(final_out, 'wb') as f:
        f.write(data_bytes)

    if verbose:
        print(f"\n✓ Arquivo recuperado: {final_out}")
        print(f"Tamanho final (bytes): {os.path.getsize(final_out)}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Decodificador compatível com FileToVideo2K")
    parser.add_argument('input_video', help='Vídeo gerado pelo encoder (mp4/avi)')
    parser.add_argument('-o', '--outdir', default='.', help='Diretório de saída')
    parser.add_argument('-b', '--block', type=int, default=DEFAULT_BLOCK_SIZE, help='Tamanho do bloco (padrão 4)')
    parser.add_argument('-s', '--sync', type=int, default=DEFAULT_SYNC_FRAMES, help='Número de frames de sincronização (padrão 2)')
    parser.add_argument('-t', '--threshold', type=int, default=128, help='Threshold de intensidade para decidir 1/0 (padrão 128)')
    parser.add_argument('-q', '--quiet', action='store_true', help='Modo silencioso')
    args = parser.parse_args()

    success = decode_video(args.input_video, output_dir=args.outdir, block_size=args.block, sync_frames=args.sync, threshold=args.threshold, verbose=not args.quiet)
    if not success:
        print("\n✗ Falha ao decodificar.")
    else:
        print("\n✓ Decodificação finalizada com sucesso.")

if __name__ == "__main__":
    main()
