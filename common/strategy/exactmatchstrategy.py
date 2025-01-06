import cv2
import sys
import os
import numpy as np
    
current_file = os.path.abspath(__file__) 
project_root = os.path.abspath(os.path.join(current_file, "..", "..", ".."))
sys.path.append(project_root)

from manage import PathManager

path_manager = PathManager()
sys.path.append(path_manager.get_path("utils"))
sys.path.append(path_manager.get_path("logs"))
sys.path.append(path_manager.get_path("common"))

try:
    from utils import capture_screen
    from logs import log_manager
except Exception as e:
    print(f"임포트 실패: {e}")

class ExactMatchStrategy:
    """
    정확한(단순) 템플릿 매칭을 수행하는 전략 클래스.
    """
    def match(self, screen_image, template_image):
            """
            템플릿 매칭을 수행합니다.

            Args:
                screen_image (str or numpy.ndarray): 화면 이미지 경로 또는 numpy 배열.
                template_image (str): 템플릿 이미지 파일명 (기본 이미지 경로).

            Returns:
                tuple:
                - bool: 매칭 성공 여부.
                - tuple or None: 매칭된 위치의 중심 좌표 (center_x, center_y) 또는 None.
            """
            if isinstance(screen_image, str):
                screen_image_data = cv2.imread(screen_image, cv2.IMREAD_GRAYSCALE)
            else:
                screen_image_data = screen_image

            if screen_image_data is None:
                raise ValueError(f"Screen image could not be loaded from {screen_image}")

            # 템플릿 이미지 파일 이름 변형 리스트 생성
            template_variations = [
                template_image,
                *[f"{template_image.split('.')[0]}{i}.png" for i in range(1, 3)]
            ]

            # test.png, test1.png, ... test3.png 모두 탐색해서 템플릿 매칭 진행
            for template_path in template_variations:
                template_image_data = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                print(template_path)
                if template_image_data is None:
                    log_manager.logger.warning(f"Template image could not be loaded from {template_path}, skipping...")
                    continue

                # 템플릿 매칭 수행
                result = cv2.matchTemplate(screen_image_data, template_image_data, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                # 임계값 설정
                threshold = 0.91
                if max_val >= threshold:
                    # 매칭된 영역의 중앙 좌표 계산
                    template_height, template_width = template_image_data.shape[:2]
                    center_x = max_loc[0] + template_width // 2
                    center_y = max_loc[1] + template_height // 2
                    log_manager.logger.info(f"Template matched: {template_path} with confidence {max_val}")
                    return True, (center_x, center_y)
                else:
                    log_manager.logger.info(f"템플릿 매칭 실패 - 매칭값: {max_val}, 쓰레시홀드: {threshold}")

            # 모든 템플릿 이미지에서 매칭 실패
            log_manager.logger.info("No matching template found.")
            return False, None

if __name__ == "__main__":
    from templetmatching import TemplateMatcher
    # TemplateMatcher 인스턴스 생성 (Singleton 패턴)
    matcher = TemplateMatcher()

    # 매칭 전략 설정 (Strategy 패턴)
    matcher.set_strategy(ExactMatchStrategy())

    # 현재 화면 캡처
    captured_screen_path = capture_screen()

    # 템플릿 매칭 수행 (정확한 경로 사용)
    template_path = os.path.join(os.getcwd(), "assets", "test", "test.png")  # 현재 폴더 기준 절대 경로 생성

    try:
        # 템플릿 매칭 수행
        is_match, location = matcher.match_template(captured_screen_path, template_path)

        if is_match:
            log_manager.logger.info(f"Template matched at location: {location}")
        else:
            log_manager.logger.info("Template did not match.")
    except ValueError as e:
        log_manager.logger.info(f"Error during matching: {e}")
