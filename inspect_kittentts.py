import inspect
from kittentts import KittenTTS
print('ctor', inspect.signature(KittenTTS))
print('generate', inspect.signature(KittenTTS.generate))
