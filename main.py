import cv2
import os


def find_features(img1):
    correct_matches_dct = {}
    frames_directory = 'images/sources/video_frames/'

    for frame_file in os.listdir(frames_directory):
        img2 = cv2.imread(os.path.join(frames_directory, frame_file), 0)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        correct_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                correct_matches.append([m])
                correct_matches_dct[frame_file.split('.')[0]] = len(correct_matches)

    correct_matches_dct = dict(sorted(correct_matches_dct.items(), key=lambda item: item[1], reverse=True))
    if correct_matches_dct:
        return list(correct_matches_dct.keys())[0]
    else:
        return "No matches found"


def find_contours_of_cards(image):
    blurred = cv2.GaussianBlur(image, (7, 7), 0)
    # T, thresh_img = cv2.threshold(blurred, 25, 230, cv2.THRESH_BINARY)
    thresh_img = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # thresh_img = cv2.Canny(blurred, 55, 200)
    # cnts, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts, _ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # cv2.imshow('mask', thresh_img)
    # cv2.waitKey(0)
    return cnts


# обираємо другий за розміром контур, інакше обираємо перший
def find_coordinates_of_cards(cnts, image):
    largest_contour = None
    second_largest_contour = None
    largest_area = 0
    second_largest_area = 0

    for i in range(0, len(cnts)):
        x, y, w, h = cv2.boundingRect(cnts[i])
        if w > 20 and h > 30:
            area = w * h
            if area > largest_area:
                second_largest_area = largest_area
                second_largest_contour = largest_contour

                largest_area = area
                largest_contour = (x - 15, y - 15, x + w + 15, y + h + 15)
            elif area > second_largest_area:
                second_largest_area = area
                second_largest_contour = (x - 15, y - 15, x + w + 15, y + h + 15)

    if second_largest_contour:
        return {'Second Largest Contour': second_largest_contour}
    elif largest_contour:
        return {'Largest Contour': largest_contour}
    else:
        return {'No matches found': None}

# Інші версії
'''
# def find_coordinates_of_cards(cnts, image):
#     cards_coordinates = {}
#     for i in range(0, len(cnts)):
#         x, y, w, h = cv2.boundingRect(cnts[i])
#         if w > 20 and h > 30:
#             img_crop = image[y - 15:y + h + 15, x - 15:x + w + 15]
#             cards_name = find_features(img_crop)
#             cards_coordinates[cards_name] = (x - 15, y - 15, x + w + 15, y + h + 15)
#
#     return cards_coordinates


# Після знайденого першого контуру цикл завершено
# def find_coordinates_of_cards(cnts, image):
#     cards_coordinates = {}
#     for i in range(0, len(cnts)):
#         x, y, w, h = cv2.boundingRect(cnts[i])
#         if w > 20 and h > 30:
#             img_crop = image[y - 15:y + h + 15, x - 15:x + w + 15]
#             cards_name = find_features(img_crop)
#             if cards_name:
#                 cards_coordinates[cards_name] = (x - 15, y - 15, x + w + 15, y + h + 15)
#                 break
#
#     return cards_coordinates


# найбільший контур
# def find_coordinates_of_cards(cnts, image):
#     best_match = None
#     max_area = 0
#
#     for i in range(0, len(cnts)):
#         x, y, w, h = cv2.boundingRect(cnts[i])
#         if w > 20 and h > 30:
#             area = w * h
#             if area > max_area:
#                 img_crop = image[y - 15:y + h + 15, x - 15:x + w + 15]
#                 cards_name = find_features(img_crop)
#                 if cards_name:
#                     best_match = {cards_name: (x - 15, y - 15, x + w + 15, y + h + 15)}
#                     max_area = area
#
#     return best_match if best_match else {'No matches found': None}
'''

def draw_rectangle_aroud_cards(cards_coordinates, image):
    for key, value in cards_coordinates.items():
        rec = cv2.rectangle(image, (value[0], value[1]), (value[2], value[3]), (255, 255, 0), 2)
        cv2.putText(rec, key, (value[0], value[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

    cv2.imshow('Image', image)
    cv2.waitKey(0)


def normalize_img(main_img):
    max_width = 900
    max_height = 800

    current_height, current_width, _ = main_img.shape

    if current_width > max_width or current_height > max_height:
        if current_width / max_width > current_height / max_height:
            new_width = max_width
            new_height = int(current_height * (max_width / current_width))
            main_image = cv2.resize(main_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        else:
            new_height = max_height
            new_width = int(current_width * (max_height / current_height))
            main_image = cv2.resize(main_img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    else:
        main_image = cv2.resize(main_img, (current_width,current_height), interpolation=cv2.INTER_NEAREST)
    return main_image


if __name__ == '__main__':
    video_path = 'images/sources/box.mp4'
    output_directory = 'images/sources/video_frames/'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

        cap = cv2.VideoCapture(video_path)

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_file = os.path.join(output_directory, f'frame_{frame_number:04d}.jpg')
            cv2.imwrite(frame_file, frame)
            frame_number += 1

        cap.release()

    # main_img - фото на якому шукаємо об'єкт
    main_img = cv2.imread('images/for_recognition/photo_5460974982298455552_y.jpg')
    main_image = normalize_img(main_img)
    gray_main_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
    contours = find_contours_of_cards(gray_main_image)
    cards_location = find_coordinates_of_cards(contours, gray_main_image)
    draw_rectangle_aroud_cards(cards_location, main_image)