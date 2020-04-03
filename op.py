import cv2
import dlib
import numpy as np

#使う画像のパス
image_path = "rin.jpg"

#画像の読み込み
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#顔検出のための宣言
detector = dlib.get_frontal_face_detector()

#学習済み分類器のデータ
cascade_path = "haarcascade_frontalface_default.xml"

#顔の特徴を検出するのに使うデータ
PREDICTOR_PATH = './shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(PREDICTOR_PATH)

#顔検出の処理を軽くするためにグレイ変換
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#顔の検出
cascade = cv2.CascadeClassifier(cascade_path)
facerect = cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=2, minSize=(200, 200))

#検出した顔を格納
image_right = image[facerect[0][1] : facerect[0][1]+facerect[0][3], facerect[0][0]: facerect[0][0]+facerect[0][2]]
image_left = image[facerect[1][1] : facerect[1][1]+facerect[1][3], facerect[1][0]: facerect[1][0]+facerect[1][2]]

#それそれの特徴点の格納場所
LEFT_EYE_POINTS = list(range(42, 48))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_BROW_POINTS = list(range(17, 22))
NOSE_POINTS = list(range(27, 35))
MOUTH_POINTS = list(range(48, 61))

#マスクの位置、サイズを調整する時に使う特徴点
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

#マスクを作成する時に使う特徴点
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

def re_land(im):
    """
     特徴点(landmarks)の座標の取得
     im: 顔の画像
    """
    # 入力画像に対して顔検出を行う
    rects = detector(im, 1)

    # 検出された顔分、特徴点抽出を行う
    for rect in rects:
        landmarks = np.matrix(
            [[p.x, p.y] for p in predictor(im, rect).parts()]
        )
    return landmarks

def transformation_from_points(t_points, o_points):
    """
      特徴点から回転やスケールを調整する。
      t_points: (target points) 対象の特徴点(マスクを被せたい画像)
      o_points: (origin points) 合成元の特徴点(マスクを作りたい画像)
    """

    #astype()による型変換
    t_points = t_points.astype(np.float64)
    o_points = o_points.astype(np.float64)

    #平均を求める
    t_mean = np.mean(t_points, axis = 0)
    o_mean = np.mean(o_points, axis = 0)

    #各要素と平均との差をとる
    t_points -= t_mean
    o_points -= o_mean

    #標準偏差を求める
    t_std = np.std(t_points)
    o_std = np.std(o_points)

    #各要素と標準偏差の差をとる
    t_points -= t_std
    o_points -= o_std

    #特異値分解を行う
    U, S, Vt = np.linalg.svd(t_points.T * o_points)
    R = (U * Vt).T

    #アフィン変換行列を返す
    return np.vstack(
      [np.hstack((( o_std / t_std ) * R, o_mean.T - ( o_std / t_std ) * R * t_mean.T )),
      np.matrix([ 0., 0., 1. ])]
    )

def warp_im(im, M, dshape):
    """
    入力画像を調整して返す
    im: 調整したい画像
    M: アフィン変換行列
    dshape: 被せたい顔の画像サイズ
    """

    #入力画像と同じサイズのゼロ行列を作る
    output_im = np.zeros(dshape, dtype=im.dtype)

    #入力画像をアフィン変換し保存
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)

    return output_im

def draw_convex_hull(im, points, color):
  """
  輪郭の凸性の欠陥（へこみなど）を修正し、塗りつぶされた凸ポリゴンを描く
  im:　元画像
  points:　特徴点の座標
  color:　塗り潰したい色
  """

  #輪郭の凸性の欠陥（へこみなど）を修正する
  points = cv2.convexHull(points)

  #頂点座標を元に入力画像に対して塗りつぶされた凸ポリゴンを描く
  cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks):
    """
    特徴部分のマスクを作成する
    im: マスクを作成するための元画像
    landmarks: 特徴点の座標
    """

    #入力画像と同じサイズのゼロ行列を作る
    im = np.zeros(im.shape[:2], im.dtype)

    #特徴点の行列を作成する
    poly = np.array(landmarks,np.int32)

    #任意の特徴領域のマスクを作成する
    for group in OVERLAY_POINTS:
        draw_convex_hull(im,poly[group],color=(255,255,255))
    return im

#初期状態のlandmarks作成
landmarks_initial_right = re_land(image_right)
landmarks_initial_left = re_land(image_left)

#右の人の顔を使ったマスクを作るための画像調整と調整後のlandmarks取得&調整
M_right = transformation_from_points(landmarks_initial_left[ALIGN_POINTS],landmarks_initial_right[ALIGN_POINTS])
w_right = warp_im(image_right,M_right,image_left.shape)
landmarks_right = re_land(w_right)
landmarks_right[48:49,0]=landmarks_right[48:49,0]-45
landmarks_right[54:55,0]=landmarks_right[54:55,0]+45

#左の人の顔を使ったマスクを作るための画像調整と調整後のlandmarks取得&調整
M_left = transformation_from_points(landmarks_initial_right[ALIGN_POINTS],landmarks_initial_left[ALIGN_POINTS])
w_left = warp_im(image_left,M_left,image_right.shape)
landmarks_left = re_land(w_left)
landmarks_left[48:49,0]=landmarks_left[48:49,0]-85
landmarks_left[54:55,0]=landmarks_left[54:55,0]+30

#右の人のマスクを取得
w_right_mask = get_face_mask(w_right,landmarks_right)
mask_left = get_face_mask(image_left,landmarks_initial_left)

#左の人のマスクを取得
w_left_mask = get_face_mask(w_left,landmarks_left)
mask_right = get_face_mask(image_right,landmarks_initial_right)

#被せるマスクと被さる顔のマスクの最大値をとることで合成部分を確定
combined_mask_right = np.max([w_right_mask, mask_left],axis=0)
combined_mask_left = np.max([w_left_mask, mask_right],axis=0)

# マスクを被せる場所の中央の位置
center = (image_left.shape[0]//2-25, image_left.shape[1]//2+15)
center2 = (image_right.shape[0]//2-5, image_right.shape[1]//2+20)

# マスクを合成
output_left = cv2.seamlessClone(w_right, image_left, combined_mask_right, center, cv2.NORMAL_CLONE)
output_right = cv2.seamlessClone(w_left, image_right, combined_mask_left, center2, cv2.NORMAL_CLONE)

#変換した顔画像を元画像に合成
image[facerect[1][1] : facerect[1][1]+facerect[1][3], facerect[1][0]: facerect[1][0]+facerect[1][2]] = output_left
image[facerect[0][1] : facerect[0][1]+facerect[0][3], facerect[0][0]: facerect[0][0]+facerect[0][2]] = output_right

# 結果を保存
cv2.imwrite("output.jpg", image)
