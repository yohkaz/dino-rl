import numpy as np

class FrameProcessor():
    def __init__(self, simplified=True):
        self.simplified = simplified

        if simplified:
            self.init_simplified()
        else:
            self.init_deep()

    def init_simplified(self):
        self.height_levels = 4
        self.obstacle_levels = 20

    def simplified_processing(self, frame, display=False):
        width, height = frame.size

        # Find first obstacle
        state = {'dist_obstacle': self.obstacle_levels}
        width, height = frame.size
        counter = 0
        obstacle = False
        for i in range(width):
            coord = (i, 132)
            if frame.getpixel(coord) != (255, 255, 255):
                counter += 1
            else:
                counter = 0

            if counter >= 5:
                obstacle = True
                state['dist_obstacle'] = int(((i-4) * self.obstacle_levels) / width)
                if display:
                    coord = (coord[0]-4, 132)
                    frame.putpixel(coord, (255, 0, 0))
                break

        # Crop out the obstacles, ground, and score txt
        width, height = frame.size
        (left, upper, right, lower) = (0, 32, 48, height)
        img_character = frame.crop((left, upper, right, lower))
        # Find height of the character
        state['height'] = 0
        width, height = img_character.size
        character = False
        for i in range(height):
            coord = (int(width / 2) - 15, i)
            if img_character.getpixel(coord) == (83, 83, 83):
                if display:
                    img_character.putpixel(coord, (255, 0, 0))
                if not character:
                    state['height'] = int((i * (self.height_levels + 1)) / height)
                    state['height'] = (self.height_levels + 1) - state['height']
                    state['height'] -= 2
                    character = True
                break

        if display:
            print(state)
            img_character.show()
            frame.show()

        return (state['height'], state['dist_obstacle'])

    def init_deep(self):
        return 0

    def deep_processing(self, frame):
        # Crop out useless elements (score, trees very far...)
        width, height = frame.size                      # 600x150
        (left, upper, right, lower) = (0, 32, width-200, height-8)
        frame = frame.crop((left, upper, right, lower)) # 400x110

        # Clear from clouds etc..
        width, height = frame.size
        for j in range(height):
            for i in range(width):
                coord = (i, j)
                pixel = frame.getpixel(coord)
                if pixel[0] >= 186 and pixel[0] <= 237:
                    frame.putpixel(coord, (255, 255, 255))

        # Resize the frame
        frame = frame.resize((80, 80)).convert('L')     # 80x80
        # frame = frame.resize((100, 80)).convert('L')    # 100x80
        # frame = frame.resize((100, 80))                 # 100x80
        np_frame = np.array(frame).astype(np.float32)   # (width, height)=100x80 x3(RGB)

        return np_frame

    def process(self, frame):
        self.deep_processing(frame)
        if self.simplified:
            return self.simplified_processing(frame)
        else:
            return self.deep_processing(frame)

    def show_crops(self, frame):
        if self.simplified:
            self.simplified_processing(frame, True)

    def dimensions(self):
        if self.simplified:
            return (self.height_levels, self.obstacle_levels+1)
