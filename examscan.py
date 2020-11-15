import math
import re
from glob import glob
from pathlib import Path

import numpy as np
import cv2
import time

from pyzbar.pyzbar import decode as zbar_decode
from pyzbar.pyzbar import ZBarSymbol
import simpleaudio


RES_FULL_HD = (1920, 1080)
RES_4K = (3840, 2160)
A3_WIDTH_LANDSCAPE = 420
A3_HEIGHT_LANDSCAPE = 297


class ExamScan:
    #_fast_resolution = (960, 540)
    _fast_resolution = (1280, 720)

    def __init__(self, video_no, resolution=RES_4K, delay=0.150, debug=False):
        self._delay = delay
        self._debug = debug
        self._state = 'start'
        self._clicked = False
        self._student_id = None
        self._exam = None
        self._perspective_matrix = None
        self._motion_calibration_max = 0
        self._motion_calibration_start = None
        self._no_motion_detector = NoMotionDetectorNew(debug=True)
        self._qrcode_state = QRCodeState()
        self._shutter_sound = simpleaudio.WaveObject.from_wave_file('shutter.wav')
        self._ready_sound = simpleaudio.WaveObject.from_wave_file('ready.wav')
        self._scanning_sound = simpleaudio.WaveObject.from_wave_file('scanning.wav')
        self._finished_sound = simpleaudio.WaveObject.from_wave_file('finished.wav')
        self._calibrating_sound = simpleaudio.WaveObject.from_wave_file('calibrating.wav')
        self._cap = cv2.VideoCapture(video_no)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        if debug:
            cv2.namedWindow('Camera')
            cv2.namedWindow('Snapshot')

    def run(self):
        while True:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.025)
                continue
            frame_small = cv2.resize(frame, self._fast_resolution,
                                     interpolation=cv2.INTER_CUBIC)
            frame_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            self._no_motion_detector.feed_frame(frame_small)
            if self._debug:
                cv2.imshow('Camera', frame_small)
            self._qrcode_state.feed_frame(frame_small)
            if self._state == 'start':
                if self._qrcode_state.calibration_visible:
                    self._calibrating_sound.play()
                    self._state = 'calibrate'
                elif self._qrcode_state.right_visible:
                    if self._no_motion_detector.has_no_motion_since_once(self._delay):
                        if self._find_perspective(frame):
                            self._motion_calibration_start = time.monotonic()
                            self._state = 'calibrate_motion'
            elif self._state == 'calibrate_motion':
                motion_factor = self._no_motion_detector.motion_factor
                if motion_factor > self._motion_calibration_max:
                    self._motion_calibration_max = motion_factor
                if time.monotonic() > self._motion_calibration_start + 2:
                    leave_threshold = round(self._motion_calibration_max, 2) + 0.1
                    #self._no_motion_detector.set_leave_threshold(leave_threshold)
                    self._ready_sound.play()
                    self._state = 'idle'
            elif self._state == 'idle':
                if self._qrcode_state.student_id is not None:
                    self._student_id = self._qrcode_state.student_id
                    self._exam = self._qrcode_state.exam
                    self._state = 'scan_wait_noqr'
                    self._no_motion_detector.reset()
                    self._scanning_sound.play()
            elif self._state == 'scan_wait_noqr':
                if self._no_motion_detector.has_no_motion_since(self._delay):
                    if self._qrcode_state.student_id is None:
                        self._snapshot(frame)
                        self._no_motion_detector.reset()
                        self._state = 'scan'
            elif self._state == 'scan':
                if self._qrcode_state.right_visible and self._qrcode_state.left_visible:
                    self._student_id = None
                    self._finished_sound.play()
                    self._state = 'idle'
                if self._no_motion_detector.has_no_motion_since_once(self._delay, True):
                    self._snapshot(frame)
                    self._no_motion_detector.reset()
            elif self._state == 'calibrate':
                if self._no_motion_detector.has_no_motion_since_once(self._delay):
                    self._calibration_snapshot(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            print(self._state)
        self._cap.release()
        cv2.destroyAllWindows()

    def _snapshot(self, frame):
        snapshot_dir = Path('outdir') / Path(self._exam) / Path(str(self._student_id))
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        next_number = len(glob(str(snapshot_dir / '*.jpg'))) + 1
        filename = str(next_number).zfill(4) + '.jpg'
        height, width, _num_colors = frame.shape
        new_height = int(width / math.sqrt(2))
        if self._perspective_matrix is not None:
            frame = cv2.warpPerspective(frame, self._perspective_matrix,
                                        (width, new_height))
        cv2.imwrite(str(snapshot_dir / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        small_width = (width // 2, new_height // 2)
        frame_small_color = cv2.resize(frame, small_width,
                                       interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Snapshot', frame_small_color)
        self._shutter_sound.play()

    def _find_perspective(self, frame):
        inner_x = 6
        inner_y = 7
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame_gray.shape
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE |
                 cv2.CALIB_CB_FAST_CHECK)
        print('looking for chessboard')
        ret, corners = cv2.findChessboardCorners(frame_gray, (inner_x, inner_y), flags)
        print('Found', ret)
        if not ret:
            return False
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_sub = cv2.cornerSubPix(frame_gray, corners, (11, 11), (-1, -1), criteria)
        corners_sub = [(x[0][0], x[0][1]) for x in corners_sub]
        indexes = [0, 5, 36, 41]
        corners_sub = np.array([corners_sub[i] for i in indexes], dtype='float32')

        # These are the offsets for the first inner square, i.e., the
        # first corner findChessboardPattern will detect. They are
        # provided as locations from the top left corner of a A3 paper
        # in landscape with mm being the unit.
        new_width = width
        new_height = int(width / math.sqrt(2))

        x_factor = (1 / A3_WIDTH_LANDSCAPE) * new_width
        y_factor = (1 / A3_HEIGHT_LANDSCAPE) * new_height
        corners_factor = np.array([
            [x_factor, y_factor],
            [x_factor, y_factor],
            [x_factor, y_factor],
            [x_factor, y_factor]
        ])
        corners_real = np.array([
            [265.0, 128.5],
            [365.0, 128.5],
            [265.0, 248.5],
            [365.0, 248.5]], dtype='float32')
        corners_real *= corners_factor

        self._perspective_matrix = cv2.getPerspectiveTransform(corners_sub,
                                                               corners_real)
        return True

    def _calibration_snapshot(self, frame):
        calibration_dir = Path('calibration')
        calibration_dir.mkdir(exist_ok=True)
        next_number = len(glob(str(calibration_dir / "*.tiff"))) + 1
        filename = str(next_number).zfill(4) + '.tiff'
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(calibration_dir / filename), frame)
        self._shutter_sound.play()


class QRCodeState:
    def __init__(self):
        self._reset()

    def _reset(self):
        self.left_visible = False
        self.right_visible = False
        self.calibration_visible = False
        self.student_id = None
        self.exam = None

    def feed_frame(self, frame):
        self._reset()
        qrcodes = zbar_decode(frame, symbols=[ZBarSymbol.QRCODE])
        for qrcode in qrcodes:
            text = qrcode.data.decode()
            if text.startswith('l:'):
                self.left_visible = True
            elif text.startswith('r:'):
                self.right_visible = True
            elif text.startswith('c:'):
                self.calibration_visible = True
            elif text.startswith('s:'):
                match = re.match(r'^s:(\d+):(\w+)', text)
                if match:
                    self.student_id = int(match.group(1))
                    self.exam = match.group(2)

class NoMotionDetectorNew:
    def __init__(self, enter_threshold=20, leave_threshold=0.5, debug=False):
        self._enter_threshold = enter_threshold
        self._leave_threshold = leave_threshold
        self._debug = debug
        self._frame = None
        self._last_frame = None
        self._motion_history = [0] * 3
        self._motion_history_index = 0
        self._last_motion_time = None
        self._no_motion_event_seen = False
        self.motion_factor = None

    def feed_frame(self, frame):
        if self._last_motion_time is None: # ???
            self._last_motion_time = time.monotonic()
        self._last_frame = self._frame
        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
        self._frame = blurred_frame
        self._detect_motion()

    def _detect_motion(self):
        if self._last_frame is None:
            return
        diff = cv2.absdiff(self._frame, self._last_frame)
        motion_factor_new = np.sum(diff**2) / self._frame.size
        self._motion_history[self._motion_history_index] = motion_factor_new
        motion_factor = sum(self._motion_history) / len(self._motion_history)
        self._motion_history_index += 1
        self._motion_history_index %= len(self._motion_history)
        self.motion_factor = motion_factor
        if motion_factor_new > self._enter_threshold:
            self._last_motion_time = time.monotonic()
            self._had_motion = True
        if motion_factor > self._leave_threshold:
            self._last_motion_time = time.monotonic()
        if self._debug:
            height, width = diff.shape
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_pos = (5, height - 10)
            text = str(round(motion_factor, 2)) + '/' + str(round(motion_factor_new, 2))
            cv2.putText(diff, text, text_pos, font, 1, 255, 2, cv2.LINE_AA)
            cv2.imshow('Motion', diff)

    def reset(self):
        self._last_motion_time = time.monotonic()
        self._no_motion_event_seen = False
        self._had_motion = False

    def has_no_motion_since(self, delay, require_motion=False):
        if self._last_motion_time is None:
            return False
        if require_motion and not self._had_motion:
            return False
        return time.monotonic() > self._last_motion_time + delay

    def has_no_motion_since_once(self, delay, require_motion=False):
        if self._no_motion_event_seen:
            return False
        result = self.has_no_motion_since(delay, require_motion)
        if result:
            self._no_motion_event_seen = True
        return result


class NoMotionDetector:
    def __init__(self, enter_threshold=5, leave_threshold=0.5, debug=False):
        self._enter_threshold = enter_threshold
        self._leave_threshold = leave_threshold
        self._debug = debug
        self._frame = None
        self._last_frame = None
        self._last_motion = time.monotonic()
        self._last_was_motion = False
        self._seen_motion = False
        self._motion_factors = [0] * 4
        self._motion_factor_index = 0
        self.motion_factor = 0
        if debug:
            cv2.namedWindow('Motion')
            cv2.createTrackbar('enter_threshold', 'Motion', 1, 500,
                    self._cb_enter_threshold_trackbar)
            cv2.setTrackbarPos('enter_threshold', 'Motion', enter_threshold)

    def set_leave_threshold(self, value):
        self._leave_threshold = value

    def feed_frame(self, frame):
        self._last_frame = self._frame
        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
        self._frame = blurred_frame
        self._determine_motion()

    def has_no_motion_since(self, duration):
        current_time = time.monotonic()
        if self._last_motion is None:
            return False
        motionless_duration = current_time - self._last_motion
        return motionless_duration >= duration

    def reset(self):
        self._seen_motion = False

    def has_no_motion_since_once(self, duration):
        if self._seen_motion:
            return False
        no_motion = self.has_no_motion_since(duration)
        if no_motion:
            self._seen_motion = True
        return no_motion

    def _determine_motion(self):
        if self._last_frame is None:
            return
        diff = cv2.absdiff(self._frame, self._last_frame)
        motion_factor = np.sum(diff**2) / self._frame.size
        self._motion_factors[self._motion_factor_index] = motion_factor
        motion_factor = sum(self._motion_factors) / len(self._motion_factors)
        self._motion_factor_index += 1
        self._motion_factor_index %= len(self._motion_factors)
        self.motion_factor = motion_factor
        if self._debug:
            diff_show = diff[:]
            height, width = diff_show.shape
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_pos = (5, height - 10)
            text = str(round(motion_factor, 2))
            cv2.putText(diff_show, text, text_pos, font, 1, 255, 2, cv2.LINE_AA)
            cv2.imshow('Motion', diff_show)
        if self._last_was_motion:
            has_motion = motion_factor > self._leave_threshold
        else:
            has_motion = motion_factor > self._enter_threshold
        current_time = time.monotonic()
        if has_motion:
            self._last_motion = current_time
            self._seen_motion = False
        self._last_was_motion = has_motion

    def _cb_enter_threshold_trackbar(self, value):
        self._enter_threshold = value


def main():
    scan = ExamScan(0, debug=True)
    scan.run()


if __name__ == '__main__':
    main()
