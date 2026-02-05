import numpy as np
import cv2
import os
import struct
import zlib
import hashlib
from tqdm import tqdm
import itertools

class FileToVideo2K:
    def __init__(self, output_video="output_2k.mp4", fps=1, resolution=(2048, 1080), block_size=4):
        """
        Codificador 2K com blocos 4x4
        """
        self.output_video = output_video
        self.fps = fps
        self.resolution = resolution
        self.width, self.height = resolution
        self.block_size = block_size
        
        # Configurações otimizadas
        self.COLOR_BLACK = 16
        self.COLOR_WHITE = 240
        
        # Calcula capacidade
        self.blocks_x = self.width // block_size
        self.blocks_y = self.height // block_size
        self.bits_per_frame = self.blocks_x * self.blocks_y
        
        print(f"=== Codificador 2K ===")
        print(f"Resolução: {resolution[0]}x{resolution[1]}")
        print(f"Bloco: {block_size}x{block_size} pixels por bit")
        print(f"Capacidade: {self.bits_per_frame:,} bits/frame ({self.bits_per_frame//8:,} bytes)")
    
    def get_available_codec(self):
        """
        Tenta encontrar um codec disponível no sistema
        """
        # Lista de codecs para tentar (em ordem de preferência)
        codecs_to_try = [
            ('mp4v', 'MPEG-4'),      # Mais comum, geralmente funciona
            ('MJPG', 'Motion JPEG'),  # Muito compatível
            ('XVID', 'Xvid'),         # Open source
            ('DIVX', 'DivX'),         # Alternativa
            ('FMP4', 'FFmpeg MPEG-4'),# Codec FFmpeg
        ]
        
        for codec_code, codec_name in codecs_to_try:
            try:
                # Testa se o codec está disponível
                fourcc = cv2.VideoWriter_fourcc(*codec_code)
                # Cria um writer de teste
                test_writer = cv2.VideoWriter(
                    'test_temp.avi',
                    fourcc,
                    self.fps,
                    (100, 100),  # Resolução pequena para teste
                    isColor=True
                )
                
                if test_writer.isOpened():
                    test_writer.release()
                    os.remove('test_temp.avi')
                    print(f"✓ Codec disponível: {codec_name} ({codec_code})")
                    return fourcc
                
                test_writer.release()
                if os.path.exists('test_temp.avi'):
                    os.remove('test_temp.avi')
                    
            except Exception as e:
                continue
        
        # Se nenhum codec funcionar, usa MP4V como fallback
        print("⚠ Nenhum codec otimizado encontrado, usando MP4V")
        return cv2.VideoWriter_fourcc(*'mp4v')
    
    def create_video_writer(self):
        """
        Cria o VideoWriter com o codec apropriado
        """
        # Tenta encontrar o melhor codec
        fourcc = self.get_available_codec()
        
        # Cria o writer
        video_writer = cv2.VideoWriter(
            self.output_video,
            fourcc,
            self.fps,
            self.resolution,
            isColor=True
        )
        
        if not video_writer.isOpened():
            # Tenta criar como AVI se MP4 falhar
            if self.output_video.endswith('.mp4'):
                avi_output = self.output_video.replace('.mp4', '.avi')
                print(f"⚠ MP4 não suportado, tentando AVI: {avi_output}")
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # MJPG é muito compatível com AVI
                video_writer = cv2.VideoWriter(
                    avi_output,
                    fourcc,
                    self.fps,
                    self.resolution,
                    isColor=True
                )
                self.output_video = avi_output
        
        return video_writer
    
    def add_error_correction(self, data_bits, ecc_strength=0.15):
        """
        Adiciona correção de erro usando paridade
        """
        # Calcula bits de paridade
        ecc_bits_needed = int(len(data_bits) * ecc_strength)
        
        # Cria paridade simples (XOR em blocos de 32 bits)
        block_size = 32
        ecc_bits = []
        
        for i in range(0, len(data_bits), block_size):
            block = data_bits[i:i+block_size]
            if len(block) < block_size:
                block = block + [0] * (block_size - len(block))
            
            # Calcula paridade do bloco
            parity = 0
            for bit in block:
                parity ^= bit
            
            # Adiciona 4 bits de paridade por bloco
            ecc_bits.extend([parity] * 4)
        
        # Limita aos bits necessários
        ecc_bits = ecc_bits[:ecc_bits_needed]
        
        return data_bits + ecc_bits
    
    def create_advanced_header(self, filename, file_size, total_bits, num_data_frames):
        """
        Cria cabeçalho avançado
        """
        filename_bytes = filename.encode('utf-8')
        filename_len = len(filename_bytes)
        
        # Estrutura simplificada
        magic = b'2KBITS01'  # 8 bytes
        version = 1
        
        # Calcula CRC
        crc_data = struct.pack('<B', filename_len) + filename_bytes + \
                   struct.pack('<Q', file_size) + \
                   struct.pack('<Q', total_bits) + \
                   struct.pack('<I', num_data_frames)
        
        crc = zlib.crc32(crc_data) & 0xFFFFFFFF
        
        # Cabeçalho completo
        header = magic
        header += struct.pack('<B', version)
        header += struct.pack('<B', filename_len)
        header += filename_bytes
        header += struct.pack('<Q', file_size)
        header += struct.pack('<Q', total_bits)
        header += struct.pack('<I', num_data_frames)
        header += struct.pack('<I', crc)
        
        # Adiciona padding
        padding_size = (64 - (len(header) % 64)) % 64
        header += bytes([0xFF]) * padding_size  # Preenche com 0xFF
        
        return header
    
    def create_robust_block(self, bit_value, block_size):
        """
        Cria bloco 4x4 robusto
        """
        if bit_value == 1:
            # Bit 1: Quadrado branco com borda mais clara
            block = np.full((block_size, block_size), self.COLOR_WHITE, dtype=np.uint8)
            
            # Adiciona borda mais clara
            border = max(1, block_size // 4)
            block[:border, :] = 255  # Borda superior
            block[-border:, :] = 255  # Borda inferior
            block[:, :border] = 255  # Borda esquerda
            block[:, -border:] = 255  # Borda direita
            
            # Centro ligeiramente mais escuro
            if block_size > 2:
                center_start = border
                center_end = block_size - border
                if center_end > center_start:
                    block[center_start:center_end, center_start:center_end] = self.COLOR_WHITE - 20
        else:
            # Bit 0: Quadrado preto com borda mais escura
            block = np.full((block_size, block_size), self.COLOR_BLACK, dtype=np.uint8)
            
            # Adiciona borda mais escura
            border = max(1, block_size // 4)
            block[:border, :] = 0  # Borda superior
            block[-border:, :] = 0  # Borda inferior
            block[:, :border] = 0  # Borda esquerda
            block[:, -border:] = 0  # Borda direita
            
            # Centro ligeiramente mais claro
            if block_size > 2:
                center_start = border
                center_end = block_size - border
                if center_end > center_start:
                    block[center_start:center_end, center_start:center_end] = self.COLOR_BLACK + 20
        
        return block
    
    def create_sync_frame(self):
        """
        Cria frame de sincronização
        """
        frame = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Padrão de barras verticais grossas
        bar_width = 16 * self.block_size  # 16 blocos = 64 pixels
        
        for x in range(0, self.width, bar_width * 2):
            # Barra branca
            x_end = min(x + bar_width, self.width)
            frame[:, x:x_end] = 255
            
            # Próxima barra (mantém preta)
        
        # Adiciona marcadores horizontais
        marker_height = 8 * self.block_size
        frame[:marker_height, :] = 128  # Cinza no topo
        frame[-marker_height:, :] = 128  # Cinza na base
        
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    def create_data_frame(self, bits):
        """
        Cria frame de dados
        """
        frame = np.zeros((self.height, self.width), dtype=np.uint8)
        
        bit_idx = 0
        for by in range(self.blocks_y):
            for bx in range(self.blocks_x):
                if bit_idx < len(bits):
                    bit_value = bits[bit_idx]
                else:
                    bit_value = 0
                
                block = self.create_robust_block(bit_value, self.block_size)
                
                y_start = by * self.block_size
                y_end = y_start + self.block_size
                x_start = bx * self.block_size
                x_end = x_start + self.block_size
                
                frame[y_start:y_end, x_start:x_end] = block
                bit_idx += 1
        
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    def encode(self, input_file, ecc_strength=0.15):
        """
        Codifica arquivo em vídeo 2K
        """
        print(f"\nCodificando: {input_file}")
        
        # Verifica se o arquivo existe
        if not os.path.exists(input_file):
            print(f"✗ Arquivo não encontrado: {input_file}")
            return False
        
        # Lê arquivo
        filename = os.path.basename(input_file)
        try:
            with open(input_file, 'rb') as f:
                data = f.read()
        except Exception as e:
            print(f"✗ Erro ao ler arquivo: {e}")
            return False
        
        file_size = len(data)
        
        # Converte para bits
        print("Convertendo para bits...")
        data_bits = []
        for byte in tqdm(data, desc="Processando bytes"):
            data_bits.extend([int(b) for b in f'{byte:08b}'])
        
        total_data_bits = len(data_bits)
        
        # Adiciona correção de erro
        print("Adicionando correção de erro...")
        protected_bits = self.add_error_correction(data_bits, ecc_strength)
        total_protected_bits = len(protected_bits)
        
        # Calcula frames necessários
        num_data_frames = (total_protected_bits + self.bits_per_frame - 1) // self.bits_per_frame
        
        print(f"\n=== Estatísticas ===")
        print(f"Arquivo: {filename}")
        print(f"Tamanho: {file_size:,} bytes ({total_data_bits:,} bits)")
        print(f"Com ECC: {total_protected_bits:,} bits")
        print(f"Frames de dados: {num_data_frames}")
        print(f"Capacidade/frame: {self.bits_per_frame:,} bits")
        
        # Cria cabeçalho
        header = self.create_advanced_header(filename, file_size, total_data_bits, num_data_frames)
        header_bits = []
        for byte in header:
            header_bits.extend([int(b) for b in f'{byte:08b}'])
        
        header_frames = (len(header_bits) + self.bits_per_frame - 1) // self.bits_per_frame
        
        print(f"Cabeçalho: {len(header):,} bytes")
        print(f"Frames de cabeçalho: {header_frames}")
        
        # Frames de sincronização
        sync_frames = 2
        total_frames = sync_frames + header_frames + num_data_frames
        
        print(f"Frames de sincronização: {sync_frames}")
        print(f"Total de frames: {total_frames}")
        print(f"Duração: {total_frames / self.fps:.1f} segundos")
        
        # Cria vídeo
        print(f"\nCriando vídeo {self.width}x{self.height}...")
        
        # Cria VideoWriter
        video_writer = self.create_video_writer()
        
        if not video_writer.isOpened():
            print("✗ Não foi possível criar o VideoWriter")
            return False
        
        # 1. Frames de sincronização
        print("Adicionando frames de sincronização...")
        for _ in range(sync_frames):
            sync_frame = self.create_sync_frame()
            video_writer.write(sync_frame)
        
        # 2. Frames de cabeçalho
        print("Adicionando cabeçalho...")
        for i in range(header_frames):
            start = i * self.bits_per_frame
            end = min(start + self.bits_per_frame, len(header_bits))
            frame_bits = header_bits[start:end]
            
            if len(frame_bits) < self.bits_per_frame:
                frame_bits.extend([0] * (self.bits_per_frame - len(frame_bits)))
            
            frame = self.create_data_frame(frame_bits)
            video_writer.write(frame)
        
        # 3. Frames de dados
        print("Adicionando dados...")
        try:
            for i in tqdm(range(num_data_frames), desc="Gerando frames de dados"):
                start = i * self.bits_per_frame
                end = min(start + self.bits_per_frame, total_protected_bits)
                frame_bits = protected_bits[start:end]
                
                if len(frame_bits) < self.bits_per_frame:
                    frame_bits.extend([0] * (self.bits_per_frame - len(frame_bits)))
                
                frame = self.create_data_frame(frame_bits)
                video_writer.write(frame)
        except KeyboardInterrupt:
            print("\n✗ Interrompido pelo usuário")
            video_writer.release()
            if os.path.exists(self.output_video):
                os.remove(self.output_video)
            return False
        except Exception as e:
            print(f"\n✗ Erro ao gerar frames: {e}")
            video_writer.release()
            return False
        
        video_writer.release()
        
        # Verifica se o vídeo foi criado
        if os.path.exists(self.output_video):
            video_size = os.path.getsize(self.output_video)
            efficiency = (file_size / video_size) * 100
            
            print(f"\n✓ Vídeo criado: {self.output_video}")
            print(f"Tamanho do vídeo: {video_size/1024/1024:.2f} MB")
            print(f"Eficiência: {efficiency:.2f}%")
            
            # Mostra informações sobre codec
            if self.output_video.endswith('.avi'):
                print("Formato: AVI (mais compatível)")
            else:
                print("Formato: MP4")
            
            return True
        else:
            print(f"\n✗ Erro: Vídeo não foi criado")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Codificador 2K com blocos 4x4')
    parser.add_argument('input_file', help='Arquivo para codificar')
    parser.add_argument('-o', '--output', default='output_2k.mp4', help='Vídeo de saída')
    parser.add_argument('-f', '--fps', type=int, default=1, help='Frames por segundo')
    parser.add_argument('-e', '--ecc', type=float, default=0.15, help='Força ECC (0.0-0.3)')
    
    args = parser.parse_args()
    
    encoder = FileToVideo2K(
        output_video=args.output,
        fps=args.fps
    )
    
    success = encoder.encode(args.input_file, args.ecc)
    
    if success:
        print("\n✓ Codificação concluída com sucesso!")
        print(f"  Arquivo: {args.input_file}")
        print(f"  Vídeo: {encoder.output_video}")
    else:
        print("\n✗ Falha na codificação")

if __name__ == "__main__":
    main()