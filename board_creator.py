import cv2

# Define board parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.CharucoBoard((7, 5), 0.04, 0.02, aruco_dict)

# Generate and save the board image
board_image = board.generateImage((1000, 700))
cv2.imwrite("charuco_board.png", board_image)
print("Board saved as 'charuco_board.png'. Print this and mount it on a flat surface.")
