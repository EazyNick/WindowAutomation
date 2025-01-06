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

class MultiScaleMatchingStrategy:
    """
    Multi-Scale 템플릿 매칭을 수행하는 전략 클래스.
    """
    def __init__(self, scale_factors=None, threshold=0.8):
        """
        Args:
            scale_factors (list of float): 템플릿의 스케일 변환 비율 리스트 (예: [0.5, 1.0, 1.5]).
            threshold (float): 매칭 임계값 (0~1 사이 값).
        """
        self.scale_factors = scale_factors if scale_factors else [0.5, 1.0, 1.5]
        self.threshold = threshold

    def match(self, screen_image, template_image):
        """
        Multi-Scale 템플릿 매칭을 수행합니다.

        Args:
            screen_image (str or numpy.ndarray): 화면 이미지 경로 또는 numpy 배열.
            template_image (str or numpy.ndarray): 템플릿 이미지 경로 또는 numpy 배열.

        Returns:
            tuple:
            - bool: 매칭 성공 여부.
            - tuple or None: 매칭된 위치의 중심 좌표 (center_x, center_y) 또는 None.
        """
        if isinstance(screen_image, str):
            screen_image_data = cv2.imread(screen_image, cv2.IMREAD_GRAYSCALE)
        else:
            screen_image_data = screen_image

        if isinstance(template_image, str):
            template_image_data = cv2.imread(template_image, cv2.IMREAD_GRAYSCALE)
        else:
            template_image_data = template_image

        if screen_image_data is None:
            raise ValueError(f"Screen image could not be loaded from {screen_image}")
        if template_image_data is None:
            raise ValueError(f"Template image could not be loaded from {template_image}")

        # 원본 템플릿 크기 디버그 로그 추가
        original_height, original_width = template_image_data.shape[:2]
        log_manager.logger.debug(f"Original template size: {original_width}x{original_height}")

        best_match = None
        best_confidence = 0

        for scale in self.scale_factors:
            # 템플릿 크기 조정
            resized_template = cv2.resize(template_image_data, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

            # 디버그 로그: 현재 스케일의 템플릿 크기
            template_height, template_width = resized_template.shape[:2]
            log_manager.logger.debug(f"Testing scale {scale} with template size: {template_width}x{template_height}")

            # 템플릿 매칭 수행
            result = cv2.matchTemplate(screen_image_data, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # 매칭 성공 여부 판단
            if max_val >= self.threshold and max_val > best_confidence:
                center_x = max_loc[0] + template_width // 2
                center_y = max_loc[1] + template_height // 2
                best_match = (center_x, center_y)
                best_confidence = max_val

        if best_match:
            return True, best_match

        return False, None

if __name__ == "__main__":
    from templetmatching import TemplateMatcher

    # TemplateMatcher 인스턴스 생성 (Singleton 패턴)
    matcher = TemplateMatcher()

    # 매칭 전략 설정 (Multi-Scale Matching Strategy)
    matcher.set_strategy(MultiScaleMatchingStrategy(scale_factors=[0.5, 1.0, 1.5, 2.0], threshold=0.9))

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

