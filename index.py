import cv2
import mediapipe as mp

# Inicializando o MediaPipe para reconhecimento de mãos
reconhecimento_maos = mp.solutions.hands
desenho_mp = mp.solutions.drawing_utils
maos = reconhecimento_maos.Hands()

# Vincular a webcam ao Python
webcam = cv2.VideoCapture(0) # Cria a conexão com a webcam

# Verificar se a webcam foi aberta corretamente
if webcam.isOpened():
    while True:  # Loop infinito para capturar frames da webcam
        # Capturar o próximo frame da webcam
        validacao, frame = webcam.read()
        
        # Se o frame foi capturado corretamente
        if validacao:
            # Converter o frame de BGR (padrão OpenCV) para RGB (padrão MediaPipe)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Processar o reconhecimento de mãos no frame RGB
            lista_maos = maos.process(frameRGB)
            h, w, _ = frame.shape
            pontos = []
            # Se mãos foram detectadas no frame
            if lista_maos.multi_hand_landmarks:
                for mao in lista_maos.multi_hand_landmarks:
                    # Desenhar os pontos e conexões das mãos no frame
                    desenho_mp.draw_landmarks(frame, mao, reconhecimento_maos.HAND_CONNECTIONS)
                    for id, cord in enumerate(mao.landmark):
                        cx, cy = int(cord.x*w), int(cord.y*h)
                        cv2.putText(frame, str(id), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
                        pontos.append((cx, cy))
                dedos = [8, 12, 16, 20]
                contador = 0
                if mao:
                    if pontos[4][0] < pontos[3][0]:
                        contador += 1  
                    for x in dedos:
                        if pontos[x][1] < pontos[x-1][1]:
                            contador +=1
                    cv2.putText(frame, str(contador), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 3)

            
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
