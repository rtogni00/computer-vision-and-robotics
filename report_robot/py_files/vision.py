def find_centroids(img, range_lower, range_upper):
        """Takes an input images and returns a list containing the x (left to right) and y (top to bottom)"""

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, range_lower, range_upper)


        circles = cv2.bitwise_and(img, img, mask=mask)

        circles_gray = cv2.cvtColor(circles, cv2.COLOR_BGR2GRAY)
        circles_gray = cv2.GaussianBlur(circles_gray, (5, 5), 0)
        _, circles_binary_image = cv2.threshold(
            circles_gray, 100, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        circles_binary_image = cv2.morphologyEx(
            circles_binary_image, cv2.MORPH_OPEN, kernel, iterations=3)
        circles_binary_image = cv2.morphologyEx(
            circles_binary_image, cv2.MORPH_CLOSE, kernel, iterations=3)

        # each pixel in the binary image is assigned a label representing the connected component it belongs to.
        output = cv2.connectedComponentsWithStats(
            circles_binary_image, 8, cv2.CV_32S)
        # as many random colors as labels
        (numLabels, labels, stats, centroids) = output

        return numLabels, centroids, stats

def img_to_world(x_img):
        x_tilde = M_inv @ x_img
        mu = float(- C_tilde[2]/x_tilde[2])  # this solves for Z=0
        X = np.squeeze(mu * x_tilde) + C_tilde
        return X


def detect_circle(img: np.ndarray, color, lower, upper, draw_img=None):
        output = find_centroids(img, lower, upper)
        (numLabels, centroids, stats) = output

        ids = []
        world_coord_x = []
        world_coord_y = []

        if color == 'green':
            offset = 1
        else:
            offset = 0

        if numLabels > 0:
            counter = 1002+offset
            for i in range(1, numLabels):
                ids.append(counter)
                x = centroids[i, cv2.CC_STAT_LEFT]
                y = centroids[i, cv2.CC_STAT_TOP]
                if draw_img is not None:
                    left = stats[i, cv2.CC_STAT_LEFT]
                    top = stats[i, cv2.CC_STAT_TOP]
                    height = stats[i, cv2.CC_STAT_HEIGHT]
                    width = stats[i, cv2.CC_STAT_WIDTH]
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area > 400 and area < 6000:
                        draw_img = cv2.rectangle(
                            draw_img, (left, top), (left+width, top+height),  (0, 255, 0), 2)
                counter = counter+2
                x, y, _ = img_to_world([x, y, 1])
                world_coord_x.append(x)
                world_coord_y.append(y)

        return ids, world_coord_x, world_coord_y