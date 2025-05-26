#!/usr/bin/env python3  # Define que esse script deve ser executado com o interpretador Python 3

# ========== IMPORTAÇÕES ==========
import rospy                      # Biblioteca do ROS para criação de nós, tópicos, etc.
import yaml                       # Biblioteca para ler arquivos de configuração YAML
import numpy as np               # Biblioteca NumPy para operações com arrays e matemática
from pathlib import Path         # Biblioteca para manipulação de caminhos de arquivos
from threading import Lock       # Para garantir acesso seguro a recursos compartilhados entre threads
from data_handler import Turtlebot3SensorData          # Classe que sincroniza e processa dados de sensores (odometria e LIDAR)
from occupancy_grid_mapping import GridMapper          # Classe responsável por manter e atualizar o mapa de ocupação
from live_map import LiveMapVisualizer                 # Classe para visualização gráfica ao vivo do mapa e sensores

# ========== CLASSE PRINCIPAL DO SISTEMA ==========
class OccupancyGridMapping:
    def __init__(self, config_path):
        # Abre e carrega o arquivo YAML de configuração
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Instancia a classe que recebe e sincroniza dados do LIDAR e odometria
        self.data_handler = Turtlebot3SensorData(self.config, use_bag=use_bag, bag_path=bag_path)

        # Instancia a classe responsável por gerar e atualizar o mapa de ocupação
        self.grid_mapper = GridMapper(self.config)

        # Instancia o visualizador para mostrar o mapa e o robô em tempo real
        self.visualizer = LiveMapVisualizer(self.config)

        # Cria um bloqueio para evitar condições de corrida (threads acessando o mapa ao mesmo tempo)
        self.lock = Lock()

        # Lista para armazenar o histórico de poses do robô (usado para desenhar o rastro)
        self.pose_history = []
        
        # Flag que indica se o sistema está em execução
        self.running = True

    # ========== LOOP PRINCIPAL DE EXECUÇÃO ==========
    def run(self):
        # Define a taxa de atualização do loop, lida do arquivo de configuração (em Hz)
        rate = rospy.Rate(self.config['ros']['rate'])

        # Enquanto o ROS estiver rodando e o sistema não tiver sido encerrado
        while not rospy.is_shutdown() and self.running:
            # Obtém a pose (posição e orientação) e os dados do LIDAR sincronizados
            pose, scan, max_range, min_range, angle_inc, angle_min, angle_max = self.data_handler.get_latest_data()

            # Se ambos os dados estiverem disponíveis e válidos
            if pose is not None and scan is not None:
                # Bloqueia o acesso à memória compartilhada (evita conflitos)
                with self.lock:
                    # Atualiza o mapa com base na nova leitura
                    self.grid_mapper.update_map(pose, scan, max_range, min_range, angle_inc, angle_min, angle_max)

                    # Adiciona a pose atual ao histórico para desenhar o rastro do robô
                    self.pose_history.append(pose)

                    # Atualiza a visualização gráfica com o novo mapa, pose e rastro
                    self.visualizer.update(
                        pose,
                        scan,
                        self.grid_mapper,
                        min_range,
                        max_range,
                        angle_inc,
                        angle_min,
                        angle_max,
                        self.pose_history
                    )
            self.grid_mapper.publish_ros_map()
            
            # Aguarda o próximo ciclo, de acordo com a taxa definida
            rate.sleep()
            
        

    # ========== SALVAR MAPA EM ARQUIVO ==========
    def save_map(self, filename):
        # Bloqueia o acesso à memória compartilhada para salvar de forma segura
        with self.lock:
            # Salva o mapa atual em um arquivo compactado (npz)
            self.grid_mapper.save_map(filename)

    # ========== ENCERRAMENTO DO SISTEMA ==========
    def shutdown(self):
        # Marca o sistema como encerrado para interromper o loop principal
        self.running = False

# ========== EXECUÇÃO DO SCRIPT COMO PROGRAMA PRINCIPAL ==========
if __name__ == '__main__':
    try:
        # Define o caminho até o arquivo de configuração YAML na mesma pasta do script
        config_path = Path(__file__).parent / 'config.yaml'

        # Optional: switch this to True to read from rosbag
        use_bag = True
        bag_path = str(Path(__file__).parent / 'RoundTripSlow_17_junho.bag')
        
        # Cria a instância da aplicação principal com a configuração carregada
        mapper = OccupancyGridMapping(config_path)

        # Registra a função de encerramento para ser chamada quando o ROS for desligado
        rospy.on_shutdown(mapper.shutdown)

        # Inicia o loop principal da aplicação
        mapper.run()

    # Caso o ROS seja interrompido (Ctrl+C, por exemplo), ignora a exceção
    except rospy.ROSInterruptException:
        pass
