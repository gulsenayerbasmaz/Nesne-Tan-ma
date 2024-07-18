import cv2
import numpy as np

# YOLO modelini yükle
net = cv2.dnn.readNet("yolov3_custom.cfg", "yolov3_custom_last.weights")

# Sınıf etiketlerini yükle
classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# YOLO için çıkış katmanlarını al
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Kamerayı aç
cap = cv2.VideoCapture(0)

while True:
    # Bir kare al
    ret, frame = cap.read()

    # Kareyi YOLO için işle
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Sonuçları işle
    class_ids = []
    confidences = []
    boxes = []
    height, width, channels = frame.shape
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Güven eşiği
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Sınırlayıcı kutuların köşe noktalarını hesapla
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression uygula
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Sonucu çiz
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Sonucu göster
    cv2.imshow("YOLO Object Detection", frame)

    # Çıkış için 'q' tuşuna basılıp basılmadığını kontrol et
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()





