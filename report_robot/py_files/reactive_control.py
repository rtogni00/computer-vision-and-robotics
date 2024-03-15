def choose_traj(self, data):
        ids, rho, alpha = data.landmark_ids, data.landmark_rs, data.landmark_alphas
        side = 1

        if self.which_side == 'outer':
            ids = (ids % 3 == 2) & (ids < 1000) & (ids > 100)  # outer ids
            side = 1
        else:
            ids = (ids % 3 == 0) & (ids < 1000) & (ids > 100)  # inner ids follow
            side = -1

        if np.sum(ids) >= 2:
            self.speed = self.speed_cruise
            rho_ids = rho[ids]
            alpha_ids = alpha[ids]
            x = rho * np.cos(alpha)
            y = rho * np.sin(alpha)

            # Closer aruco on the left
            index = np.argsort(rho_ids)
            first = index[0]
            second = index[1]
            x_2 = x[second]
            x_1 = x[first]
            y_2 = y[second]
            y_1 = y[first]
            target_x = x_2 - x_1
            target_y = y_2 - y_1

            theta = np.rad2deg(np.arctan2(target_y, target_x))
            direction = 1 if theta >= 0 else -1
            if abs(theta) < 10:
                self.turn = 0
            elif abs(theta) < 30:
                self.turn = 10
            elif abs(theta) < 45:
                self.turn = 20
            else:
                self.turn = 80

            if abs(rho_ids[first]*np.sin(alpha_ids[first])) < 0.13 and (rho_ids[first]*np.cos(alpha_ids[first])) < 0.4:
                self.turn = int(self.turn+10)*direction * side
                print(":warning:[bright_red]Too close")

            if rho_ids[first] > 0.35:
                self.turn = int(self.turn * direction/2)
                print(":warning:[bright_red]Far arucos")
            else:
                self.turn = int(self.turn*direction)

        else:
            print(':warning:[blue]I can\'t see a thing')
            self.turn = -150*side
            self.speed = self.speed_lost