import cv2, glob
for f in glob.glob("images/original/*"):
    im = cv2.imread(f)
    s = 1600 / max(im.shape[:2])
    if s < 1:
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imwrite(f, im, [cv2.IMWRITE_JPEG_QUALITY, 95])