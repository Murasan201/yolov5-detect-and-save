import torch
import cv2
import os
import numpy as np

def load_model():
    """
    YOLOv5モデルのロード（YOLOv5sを使用）
    """
    try:
        # 環境変数 TORCH_HOME を設定
        os.environ['TORCH_HOME'] = '/home/win/torch_cache'
        # Torch Hub のキャッシュディレクトリを設定
        torch.hub.set_dir('/home/win/torch_cache/hub')

        print("YOLOv5モデルをロード中...")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)
        print("モデルのロードが完了しました。")
        return model
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")
        exit(1)

def get_image_path():
    """
    ユーザーから画像パスを取得し、存在を確認する
    """
    while True:
        image_path = input("検出対象の画像ファイルのパスを入力してください: ").strip('"').strip("'")
        if os.path.isfile(image_path):
            return image_path
        else:
            print(f"エラー: 指定された画像ファイルが存在しません: {image_path}")
            retry = input("もう一度入力しますか？ (y/n): ").strip().lower()
            if retry != 'y':
                print("プログラムを終了します。")
                exit(1)

def filter_detections(results, target_classes):
    """
    検出結果から指定したクラスのみをフィルタリングする
    """
    # COCOデータセットのクラス名を取得
    class_names = results.names

    # ターゲットクラスのIDを取得
    target_class_ids = [id for id, name in class_names.items() if name in target_classes]

    # 検出結果のフィルタリング
    if len(results.xyxy[0]) == 0:
        return None, target_class_ids

    # TensorをNumPy配列に変換
    detections = results.xyxy[0].cpu().numpy()

    # クラスIDでフィルタリング
    filtered_detections = detections[np.isin(detections[:, -1], target_class_ids)]
    return filtered_detections, target_class_ids

def draw_boxes(image, detections, class_names):
    """
    検出されたオブジェクトにバウンディングボックスとラベルを描画する
    """
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = f"{class_names[int(cls)]} {conf:.2f}"
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        # バウンディングボックスを描画
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # ラベルを描画
        cv2.putText(image, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

def save_detected_image(image_path, image):
    """
    検出結果の画像をファイルとして保存する
    """
    # 元のファイル名と拡張子を取得
    base, ext = os.path.splitext(image_path)
    # 新しいファイル名を作成（例: image_detected.jpg）
    output_path = f"{base}_detected{ext}"
    
    # 画像を保存
    try:
        cv2.imwrite(output_path, image)
        print(f"検出結果の画像を保存しました: {output_path}")
    except Exception as e:
        print(f"画像の保存中にエラーが発生しました: {e}")

def main():
    # モデルのロード
    model = load_model()

    while True:
        # ユーザーから画像パスを取得
        image_path = get_image_path()

        # 推論の実行
        print(f"画像を処理中: {image_path}")
        try:
            results = model(image_path)
        except Exception as e:
            print(f"推論中にエラーが発生しました: {e}")
            continue

        # ターゲットクラスの指定
        target_classes = ['person', 'dog']
        filtered_detections, target_class_ids = filter_detections(results, target_classes)

        # 結果の表示
        if filtered_detections is not None and len(filtered_detections):
            print(f"検出されたオブジェクト ({', '.join(target_classes)}):")
            for det in filtered_detections:
                x1, y1, x2, y2, conf, cls = det
                class_name = model.names[int(cls)]
                print(f"- {class_name} (信頼度: {conf:.2f}) 座標: ({int(x1)}, {int(y1)}), ({int(x2)}, {int(y2)})")

            # 画像の読み込みとバウンディングボックスの描画
            image = cv2.imread(image_path)
            image = draw_boxes(image, filtered_detections, model.names)

            # 検出結果の画像を保存
            save_detected_image(image_path, image)

            # 結果画像の表示
            window_name = "Detection Results"
            cv2.imshow(window_name, image)
            print("検出結果の画像を表示しています。キーを押すと閉じます。")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"指定されたクラス（{', '.join(target_classes)}）のオブジェクトは検出されませんでした。")

        # 続行確認
        cont = input("他の画像を処理しますか？ (y/n): ").strip().lower()
        if cont != 'y':
            print("プログラムを終了します。")
            break

if __name__ == "__main__":
    main()
