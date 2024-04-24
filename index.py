import cv2
import mediapipe as mp
import time

# Inicializando o MediaPipe para reconhecimento de mãos
reconhecimento_maos = mp.solutions.hands
desenho_mp = mp.solutions.drawing_utils
maos = reconhecimento_maos.Hands()

# Vincular a webcam ao Python
webcam = cv2.VideoCapture(0) # Cria a conexão com a webcam

min_pontos_movimento = 10
movimento = "Neutro"
frame_anterior = None

# Tempo de espera entre os quadros (em segundos)
tempo_espera = 0.2

# Verificar se a webcam foi aberta corretamente
if webcam.isOpened():
    while True:  # Loop infinito para capturar frames da webcam
        # Capturar o próximo frame da webcam
        validacao, frame = webcam.read()
        
        # Se o frame foi capturado corretamente
        if validacao:
            # Inverter o frame ao longo do eixo x
            frame = cv2.flip(frame, 1)
            # Converter o frame de BGR (padrão OpenCV) para RGB (padrão MediaPipe)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Processar o reconhecimento de mãos no frame RGB
            lista_maos = maos.process(frameRGB)
            h, w, _ = frame.shape
            pontos = []

            # Desenhar linhas e legendar os pixels na tela
            # for y in range(0, h, 50):
            #     cv2.line(frame, (0, y), (w, y), (0, 255, 0), 1)  # Desenha linhas horizontais
            #     for x in range(0, w, 50):
            #         cv2.line(frame, (x, 0), (x, h), (0, 255, 0), 1)  # Desenha linhas verticais
            #         cv2.putText(frame, f'({x}, {y})', (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

            # Se mãos foram detectadas no frame
            if lista_maos.multi_hand_landmarks:
                for mao in lista_maos.multi_hand_landmarks:
                    # Desenhar os pontos e conexões das mãos no frame
                    desenho_mp.draw_landmarks(frame, mao, reconhecimento_maos.HAND_CONNECTIONS)
                    for id, cord in enumerate(mao.landmark):
                        cx, cy = int(cord.x * w), int(cord.y * h)
                        cv2.putText(frame, f'{id}', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        pontos.append((cx, cy, id))

                # Exemplo de ação com base na posição da mão
                dedos = [8, 12, 16, 20]
                contador = 0
                if mao:
                    if pontos[0][0] < pontos[4][0]:
                        if pontos[4][0] > pontos[3][0]:
                            contador += 1  
                    if pontos[0][0] > pontos[4][0]:
                        if pontos[4][0] < pontos[3][0]:
                            contador += 1  
                    for posicao_dedos in dedos:
                        if pontos[posicao_dedos][1] < pontos[posicao_dedos-1][1]:
                            contador += 1
                    cv2.putText(frame, str(contador), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)

                if len(pontos) == 21:  # Verifica se todos os pontos estão presentes
                    if frame_anterior is not None:
                        # Calcula a diferença média entre as coordenadas dos pontos entre o frame atual e o frame anterior
                        dif_x = sum([p[0] - frame_anterior[i][0] for i, p in enumerate(pontos)]) / len(pontos)
                        dif_y = sum([p[1] - frame_anterior[i][1] for i, p in enumerate(pontos)]) / len(pontos)
                        movimento = "Neutro"
                        
                        if abs(dif_y) >= min_pontos_movimento:
                            if dif_y > 0:
                                movimento = "Baixo"
                            else:
                                movimento = "Cima"
                        elif abs(dif_x) >= min_pontos_movimento:
                            if dif_x > 0:
                                movimento = "Direita"
                            else:
                                movimento = "Esquerda"

                    frame_anterior = [(p[0], p[1]) for p in pontos]  # Atualiza as coordenadas do frame anterior
                    time.sleep(tempo_espera)

                # Mostra o movimento na tela
                cv2.putText(frame, movimento, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Mostrar o frame da webcam
            cv2.imshow("Video da Webcam", frame)
            
            # Aguardar um pouco antes de capturar o próximo frame
            tecla = cv2.waitKey(1)
            
            
            # Verificar se a tecla 'Esc' foi pressionada ou se a janela foi fechada pelo usuário
            if tecla == 27 or cv2.getWindowProperty('Video da Webcam', cv2.WND_PROP_VISIBLE) < 1:
                break

# Liberar os recursos da webcam e fechar todas as janelas
webcam.release()
cv2.destroyAllWindows()
