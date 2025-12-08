from src.data.schema import GuitarSample

# Create a sample based on a song you know!
my_sample = GuitarSample(
    id="my_first_sample",
    prompt="song should be happy, with love and fast paced",
    chords=["D", "G", "C", "Em"],  # Put your chords
    strum_pattern="DDDU_UD_",       # 8 characters
    tempo=100,
    genre="pop",
    emotion="melancholic",
    key="C",
    mode="major"
)

print(my_sample)