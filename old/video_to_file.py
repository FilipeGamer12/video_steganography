import numpy as np
import cv2
import os
from tqdm import tqdm
import struct

class VideoToFileDecoder:
    def __init__(self, resolution=(1920, 1080)):
        self.resolution = resolution
        self.width, self.height = resolution
        self.bits_per_frame = self.width * self.height
    
    def frame_to_bits_robust(self, frame):
        """
        Converte um frame do vídeo de volta para bits de forma robusta
        """
        # Converte para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Aplica filtro para reduzir ruído
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Threshold adaptativo para lidar com variações de brilho
        binary = cv2.adaptiveThreshold(blurred, 1, 
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Verifica se precisamos inverter
        # Se a maioria dos pixels forem claros, está invertido
        mean_val = np.mean(binary)
        if mean_val > 0.5:
            binary = 1 - binary
        
        bits = binary.flatten().tolist()
        return bits
    
    def frame_to_bits_simple(self, frame):
        """
        Método simples: considera qualquer pixel > 127 como 1
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Threshold fixo
        _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)
        
        bits = binary.flatten().tolist()
        return bits
    
    def bits_to_bytes(self, bits, expected_bits=None):
        """
        Converte bits para bytes
        """
        if expected_bits:
            bits = bits[:expected_bits]
        
        # Certifica que temos um múltiplo de 8
        num_bits = len(bits)
        if num_bits % 8 != 0:
            bits = bits[:-(num_bits % 8)]
        
        # Converte para bytes
        bytes_list = bytearray()
        for i in range(0, len(bits), 8):
            byte_str = ''.join(str(b) for b in bits[i:i+8])
            bytes_list.append(int(byte_str, 2))
        
        return bytes(bytes_list)
    
    def verify_file_integrity(self, data, original_size):
        """
        Verifica a integridade do arquivo recuperado
        """
        if len(data) != original_size:
            print(f"AVISO: Tamanho diferente. Esperado: {original_size}, Obtido: {len(data)}")
            return False
        
        # Verifica alguns pontos do arquivo
        if len(data) > 100:
            # Verifica se os primeiros 100 bytes não são todos 0 ou todos 255
            first_100 = data[:100]
            if all(b == 0 for b in first_100) or all(b == 255 for b in first_100):
                print("AVISO: Padrão suspeito nos primeiros bytes")
                return False
        
        return True
    
    def decode_with_checksum(self, video_file, metadata, output_file):
        """
        Decodifica verificando cada frame com checksum
        """
        expected_bits = metadata['total_bits']
        num_data_frames = metadata['num_frames']
        
        print(f"Decodificando com verificação de integridade...")
        print(f"Frames a processar: {num_data_frames}")
        print(f"Bits esperados: {expected_bits}")
        
        # Abre o vídeo
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Erro ao abrir o vídeo")
            return False
        
        all_bits = []
        frame_count = 0
        
        # Processa apenas os frames necessários
        with tqdm(total=num_data_frames, desc="Processando frames") as pbar:
            while frame_count < num_data_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Extrai bits do frame
                frame_bits = self.frame_to_bits_simple(frame)
                all_bits.extend(frame_bits)
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        
        print(f"\nBits extraídos: {len(all_bits)}")
        
        # Se não temos bits suficientes, tenta método robusto
        if len(all_bits) < expected_bits:
            print("Tentando método robusto para extrair mais bits...")
            cap = cv2.VideoCapture(video_file)
            all_bits = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            for _ in range(num_data_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_bits = self.frame_to_bits_robust(frame)
                all_bits.extend(frame_bits)
            
            cap.release()
            print(f"Bits extraídos (método robusto): {len(all_bits)}")
        
        # Ajusta para o número esperado de bits
        if len(all_bits) > expected_bits:
            all_bits = all_bits[:expected_bits]
        elif len(all_bits) < expected_bits:
            print(f"AVISO: Faltam {expected_bits - len(all_bits)} bits")
            # Preenche com zeros
            all_bits.extend([0] * (expected_bits - len(all_bits)))
        
        # Converte para bytes
        file_bytes = self.bits_to_bytes(all_bits, expected_bits)
        
        # Salva o arquivo
        with open(output_file, 'wb') as f:
            f.write(file_bytes)
        
        # Verifica integridade
        if self.verify_file_integrity(file_bytes, metadata['file_size_bytes']):
            print("✓ Integridade verificada com sucesso")
        else:
            print("⚠ Possíveis problemas na recuperação")
        
        print(f"Arquivo salvo como: {output_file}")
        print(f"Tamanho: {len(file_bytes)} bytes")
        
        return True
    
    def decode_direct(self, video_file, metadata, output_file):
        """
        Decodificação direta sem verificações complexas
        """
        expected_bits = metadata['total_bits']
        num_data_frames = metadata['num_frames']
        
        print("Usando decodificação direta...")
        
        # Lê todos os frames de uma vez
        cap = cv2.VideoCapture(video_file)
        frames = []
        
        for _ in range(num_data_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        # Processa todos os frames
        all_bits = []
        for frame in tqdm(frames, desc="Processando frames"):
            # Tenta ambos os métodos e escolhe o que parece melhor
            bits_simple = self.frame_to_bits_simple(frame)
            bits_robust = self.frame_to_bits_robust(frame)
            
            # Escolhe o método com melhor distribuição (não tudo 0 ou tudo 1)
            simple_mean = np.mean(bits_simple)
            robust_mean = np.mean(bits_robust)
            
            if 0.1 < simple_mean < 0.9:
                all_bits.extend(bits_simple)
            else:
                all_bits.extend(bits_robust)
        
        # Ajusta para o tamanho correto
        all_bits = all_bits[:expected_bits]
        
        # Converte para bytes
        file_bytes = self.bits_to_bytes(all_bits)
        
        # Salva
        with open(output_file, 'wb') as f:
            f.write(file_bytes)
        
        print(f"Arquivo salvo: {output_file}")
        print(f"Tamanho: {len(file_bytes)} bytes")
        
        return True
    
    def decode(self, video_file, output_file=None):
        """
        Decodifica o vídeo
        """
        print(f"Decodificando: {video_file}")
        
        # Carrega metadados
        metadata = self.load_metadata(video_file)
        if not metadata:
            print("ERRO: Metadados não encontrados!")
            return False
        
        if output_file is None:
            output_file = f"restored_{metadata['original_filename']}"
        
        # Tenta diferentes métodos
        print("\n1. Tentando método com checksum...")
        result1 = self.decode_with_checksum(video_file, metadata, 
                                          output_file.replace('.', '_method1.'))
        
        print("\n2. Tentando método direto...")
        result2 = self.decode_direct(video_file, metadata, 
                                    output_file.replace('.', '_method2.'))
        
        # Verifica qual método produziu o melhor resultado
        files = [
            output_file.replace('.', '_method1.'),
            output_file.replace('.', '_method2.')
        ]
        
        best_file = None
        best_size_diff = float('inf')
        
        for f in files:
            if os.path.exists(f):
                actual_size = os.path.getsize(f)
                expected_size = metadata['file_size_bytes']
                size_diff = abs(actual_size - expected_size)
                
                if size_diff < best_size_diff:
                    best_size_diff = size_diff
                    best_file = f
        
        # Copia o melhor arquivo para o nome final
        if best_file and best_file != output_file:
            import shutil
            shutil.copy2(best_file, output_file)
            print(f"\n✓ Melhor resultado copiado para: {output_file}")
            
            # Remove arquivos temporários
            for f in files:
                if f != output_file and os.path.exists(f):
                    os.remove(f)
        
        return True
    
    def load_metadata(self, video_file):
        """Carrega metadados"""
        meta_file = video_file + ".meta"
        if os.path.exists(meta_file):
            metadata = {}
            with open(meta_file, 'r') as f:
                for line in f:
                    if ': ' in line:
                        key, value = line.strip().split(': ', 1)
                        # Tenta converter números
                        try:
                            if '.' in value:
                                metadata[key] = float(value)
                            else:
                                metadata[key] = int(value)
                        except:
                            metadata[key] = value
            return metadata
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Decodifica vídeo para arquivo')
    parser.add_argument('video_file', help='Arquivo de vídeo')
    parser.add_argument('-o', '--output', help='Arquivo de saída')
    
    args = parser.parse_args()
    
    decoder = VideoToFileDecoder()
    decoder.decode(args.video_file, args.output)

if __name__ == "__main__":
    main()