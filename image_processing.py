import cv2

try:
    import imutils
except ImportError:  # pragma: no cover - only needed for standalone demo execution
    imutils = None

from modules.extract import ellipse_fit, erode_thresh, extract_circles


def main(image_path="owl1.jpg"):
    if imutils is None:
        raise ImportError("imutils is required to run image_processing.py")

    test_img = cv2.imread(image_path)
    if test_img is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    circle = extract_circles(test_img)
    cv2.imshow("extracted circle", imutils.resize(circle, width=432, height=324))

    threshed_image = erode_thresh(circle)
    cv2.imshow(
        "eroded and threshed",
        imutils.resize(threshed_image, width=432, height=324),
    )

    final_image = ellipse_fit(circle, threshed_image)
    cv2.imshow("window", imutils.resize(final_image, width=432, height=324))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
