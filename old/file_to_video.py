import numpy as np
import cv2
import os
from tqdm import tqdm

class FileToVideoEncoder:
    def __init__(self, output_video="output.mp4", fps=1, resolution=(1920, 1080)):
        """
        Inicializa o codificador
        
        Args:
            output_video: Nome do arquivo de vídeo de saída
            fps: Frames por segundo (padrão: 1 fps = 1 frame por segundo)
            resolution: Resolução do vídeo (padrão: 1080p)
        """
        self.output_video = output_video
        self.fps = fps
        self.resolution = resolution
        self.width, self.height = resolution
        
        # Total de bits por frame
        self.bits_per_frame = self.width * self.height
        
    def file_to_bits(self, file_path):
        """
        Lê um arquivo e converte para uma sequência de bits
        
        Args:
            file_path: Caminho do arquivo a ser codificado
            
        Returns:
            Lista de bits (0s e 1s)
        """
        with open(file_path, 'rb') as file:
            data = file.read()
        
        # Converte bytes para string de bits
        bits = []
        for byte in data:
            # Converte byte para string binária de 8 bits, preenche com zeros à esquerda
            bits.extend([int(b) for b in f'{byte:08b}'])
        
        return bits, len(data)
    
    def create_frame(self, bits, frame_index):
        """
        Cria um frame do vídeo a partir de uma sequência de bits
        
        Args:
            bits: Lista de todos os bits
            frame_index: Índice do frame atual
            
        Returns:
            Frame como array numpy
        """
        # Calcula os bits para este frame específico
        start_idx = frame_index * self.bits_per_frame
        end_idx = min(start_idx + self.bits_per_frame, len(bits))
        
        # Cria uma matriz 2D com os bits
        frame_bits = bits[start_idx:end_idx]
        
        # Se não tiver bits suficientes para preencher o frame, preenche com zeros
        if len(frame_bits) < self.bits_per_frame:
            frame_bits.extend([0] * (self.bits_per_frame - len(frame_bits)))
        
        # Converte para array 2D
        bit_matrix = np.array(frame_bits, dtype=np.uint8).reshape(self.height, self.width)
        
        # Converte 0s e 1s para pixels preto (0) e branco (255)
        frame = bit_matrix * 255
        
        # Converte para 3 canais (BGR para OpenCV)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        return frame
    
    def encode(self, input_file):
        """
        Codifica um arquivo em vídeo
        
        Args:
            input_file: Caminho do arquivo a ser codificado
        """
        print(f"Codificando arquivo: {input_file}")
        
        # Lê o arquivo e converte para bits
        bits, file_size = self.file_to_bits(input_file)
        total_bits = len(bits)
        
        # Calcula o número de frames necessários
        num_frames = (total_bits + self.bits_per_frame - 1) // self.bits_per_frame
        
        print(f"Tamanho do arquivo: {file_size} bytes ({total_bits} bits)")
        print(f"Bits por frame: {self.bits_per_frame}")
        print(f"Frames necessários: {num_frames}")
        print(f"Duração do vídeo: {num_frames / self.fps:.2f} segundos")
        
        # Cria o objeto VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            self.output_video,
            fourcc,
            self.fps,
            self.resolution
        )
        
        # Cria e salva cada frame
        for frame_idx in tqdm(range(num_frames), desc="Criando frames"):
            frame = self.create_frame(bits, frame_idx)
            video_writer.write(frame)
        
        # Adiciona um frame final de marcação (opcional)
        # Podemos adicionar um frame com padrão especial para indicar fim
        marker_frame = self.create_marker_frame()
        video_writer.write(marker_frame)
        
        video_writer.release()
        print(f"Vídeo salvo como: {self.output_video}")
        
        # Salva metadados em um arquivo de texto
        self.save_metadata(input_file, file_size, total_bits, num_frames)
        
    def create_marker_frame(self):
        """
        Cria um frame de marcação para indicar o final dos dados
        """
        # Cria um padrão alternado para identificar o frame final
        marker = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Padrão de xadrez para fácil identificação
        for i in range(self.height):
            for j in range(self.width):
                if (i // 8 + j // 8) % 2 == 0:
                    marker[i, j] = 255
        
        frame = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        return frame
    
    def save_metadata(self, input_file, file_size, total_bits, num_frames):
        """
        Salva metadados do arquivo original
        """
        metadata = {
            'original_filename': os.path.basename(input_file),
            'file_size_bytes': file_size,
            'total_bits': total_bits,
            'num_frames': num_frames,
            'resolution': self.resolution,
            'fps': self.fps,
            'bits_per_frame': self.bits_per_frame
        }
        
        with open(f"{self.output_video}.meta", 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        print(f"Metadados salvos em: {self.output_video}.meta")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Codifica um arquivo em vídeo')
    parser.add_argument('input_file', help='Arquivo a ser codificado')
    parser.add_argument('-o', '--output', default='output.mp4', help='Nome do vídeo de saída')
    parser.add_argument('-f', '--fps', type=int, default=1, help='Frames por segundo')
    
    args = parser.parse_args()
    
    encoder = FileToVideoEncoder(output_video=args.output, fps=args.fps)
    encoder.encode(args.input_file)

if __name__ == "__main__":
    main()