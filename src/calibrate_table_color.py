"""
Ferramenta interativa: clique na mesa e no fundo para ver os valores HSV reais.
Uso:
    python src/calibrate_table_color.py --video data/test6/test6.mp4 --frame 50
"""
import argparse
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, default=50)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        print("Erro ao ler o frame.")
        return

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    vis = frame.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            bgr = frame[y, x]
            h, s, v = hsv[y, x]
            print(f"Pos ({x}, {y}) -> BGR: {bgr.tolist()} | HSV: [H={h}, S={s}, V={v}]")
            cv2.circle(vis, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(vis, f"H:{h} S:{s} V:{v}", (x + 8, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.imshow("Calibracao Mesa (Clique para inspecionar / ESC para sair)", vis)

    cv2.namedWindow("Calibracao Mesa (Clique para inspecionar / ESC para sair)")
    cv2.setMouseCallback("Calibracao Mesa (Clique para inspecionar / ESC para sair)", on_mouse)
    cv2.imshow("Calibracao Mesa (Clique para inspecionar / ESC para sair)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()