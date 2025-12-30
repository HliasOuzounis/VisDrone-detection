import pandas as pd
from enum import IntEnum
from typing import Tuple

class VisDroneClasses(IntEnum):
    IGNORED = 0
    PEDESTRIAN = 1
    PERSON = 2
    BICYCLE = 3
    CAR = 4
    VAN = 5
    TRUCK = 6
    TRICYCLE = 7
    AWNING_TRICYCLE = 8
    BUS = 9
    MOTOR = 10
    OTHERS = 11

class Annotations:
    def __init__(self, annot_path: str):
        self.path = annot_path
        # Standard MOT columns
        self.columns = [
            'frame', 'id', 'x', 'y', 'w', 'h', 
            'score', 'class_id', 'truncation', 'occlusion'
        ]
        self._data = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.path, header=None, names=self.columns)
        
        df = df[df['class_id'] != VisDroneClasses.IGNORED]
        
        df['class_id'] = df['class_id'].astype(int)
        df['class_name'] = df['class_id'].apply(lambda x: VisDroneClasses(x).name) # For readability
        
        return df

    def get_frame(self, frame_id: int) -> pd.DataFrame:
        """Returns all objects in a specific frame as a list of dicts."""
        frame_data = self._data[self._data['frame'] == frame_id]
        return frame_data.reset_index(drop=True)

    def get_frame_range(self) -> Tuple[int, int]:
        return (
            self._data['frame'].min(),
            self._data['frame'].max()
        )
        
    def get_object_trajectory(self, obj_id: int) -> pd.DataFrame:
        """Returns the entire history of a specific object (for tracking)."""
        return self._data[self._data['id'] == obj_id].sort_values('frame')

if __name__ == '__main__':
    example_path = '../data/VisDrone2019-VID-test-dev/annotations/uav0000009_03358_v.txt'
    annot_example = Annotations(example_path)

    min_frame, max_frame = annot_example.get_frame_range()
    print(f'{max_frame - min_frame + 1} frames total from frame {min_frame} to {max_frame}.')

    frame_id = 10
    frame_data = annot_example.get_frame(frame_id)
    print(f'Frame {frame_id} has {len(frame_data)} objects.')
    print(frame_data)