import time
from typing import List
from dataclasses import dataclass, asdict, field
from config import get_config

@dataclass
class GenerateImageRequest:
    prompt: str
    n: int = 1
    quality: str = 'hd'
    size: str = '512x512'
    model: str = ''
    response_format: str = 'b64_json'

    negative_prompt: str = ''
    guidance_scale: float = 7.5

    def validate(self):
        if self.model and self.model != get_config().model:
            raise ValueError(f'model {self.model} not found')
        ps = self.size.split('x')
        if len(ps) != 2:
            raise ValueError(f'{self.size} is not a valid size')
        if int(ps[0]) % 8 != 0 or int(ps[1]) % 8 != 0:
            raise ValueError(f'size width or height must be multiples of 8')
        if self.quality not in ('hd',):
            raise ValueError(f'{self.quality} is not a valid quality')
        assert self.num_inference_steps >= 1, "num_inference_steps must be greater than or equal to 1"
        assert 1 <= self.n <= 9, "n must be between 1 and 9"
        # todo impl url
        assert self.response_format in ('b64_json', ), "response_format must be 'b64_json' or 'url'"

    @property
    def num_inference_steps(self):
        m = {
            'hd': 50
        }
        return m[self.quality]

    @property
    def width(self):
        ps = self.size.split('x')
        return int(ps[0])

    @property
    def height(self):
        ps = self.size.split('x')
        return int(ps[1])


@dataclass
class Image:
    b64_json: str = None
    url: str = None
    revised_prompt: str = None


@dataclass
class GenerateImageResponse:
    data: List[Image]
    created: int = field(default_factory=lambda: time.time())

    def as_dict(self):
        return asdict(self)
