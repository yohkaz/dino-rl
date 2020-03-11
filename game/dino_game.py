from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

class DinoGame:
    def __init__(self, accelerate=True):
        print("DinoGame/__init__")

        # Define options
        options = Options()
        options.add_argument("--disable-infobars")
        options.add_argument("--mute-audio")
        options.add_argument('--no-sandbox')

        # Open browser and the game
        print("     Running chrome")
        self.browser = webdriver.Chrome(executable_path="chromedriver/chromedriver.exe", options=options)
        print("     Opening \'chrome://dino\'")
        self.browser.get("chrome://dino")
        print()
        if accelerate:
            self.set_parameter('config.ACCELERATION', 0)

    def get_parameters(self):
        params = {}
        params['config.ACCELERATION'] = self.browser.execute_script('return Runner.config.ACCELERATION;')
        return params

    def is_crashed(self):
        return self.browser.execute_script('return Runner.instance_.crashed;')

    def is_inverted(self):
        return self.browser.execute_script('return Runner.instance_.inverted;')

    def is_paused(self):
        return self.browser.execute_script('return Runner.instance_.paused;')

    def is_playing(self):
        return self.browser.execute_script('return Runner.instance_.playing;')

    def press_space(self):
        return self.browser.find_element_by_tag_name('body').send_keys(Keys.SPACE)

    def press_up(self):
        return self.browser.find_element_by_tag_name('body').send_keys(Keys.UP)

    def press_down(self):
        return self.browser.find_element_by_tag_name('body').send_keys(Keys.DOWN)

    def pause(self):
        return self.browser.execute_script('Runner.instance_.stop();')

    def resume(self):
        return self.browser.execute_script('Runner.instance_.play();')

    def restart(self):
        return self.browser.execute_script('Runner.instance_.restart();')

    def close(self):
        self.browser.close()

    def get_score(self):
        digits = self.browser.execute_script('return Runner.instance_.distanceMeter.digits;')
        return int(''.join(digits))

    def get_canvas(self):
        return self.browser.execute_script('return document.getElementsByClassName("runner-canvas")[0].toDataURL().substring(22);')

    def set_parameter(self, key, value):
        self.browser.execute_script('Runner.{} = {};'.format(key, value))

    def restore_parameter(self, key):
        self.set_parameter(key, self.defaults[key])
