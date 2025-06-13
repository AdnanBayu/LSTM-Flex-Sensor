import pygame
import time

class AudioPlayer:
    def __init__(self, volume=1, time_sleep=0.1):
        """Initialize the audio player and its mixer."""
        self.mixer = pygame.mixer
        self.mixer.init()
        print("Audio Player initialized.")
        
        # Set the mixer volume
        self.mixer.music.set_volume(volume)  # Set the volume to 50%
        
        # Set the time_sleep
        self.time_sleep = time_sleep

    def play_audio(self, audio_location):
        """
        Play the audio file at the specified location.
        
        Args:
            audio_location (str): Path to the audio file.
        """
        try:
            # Load and play the audio file
            self.mixer.music.load(audio_location)
            self.mixer.music.play()
            print(f"Playing: {audio_location}")

            # Wait for the audio to finish
            while self.mixer.music.get_busy():
                time.sleep(self.time_sleep)  # Prevent busy-waiting
            print("Playback finished.")

        except pygame.error as e:
            print(f"Error playing audio: {e}")


# Example usage
if __name__ == "__main__":
    player = AudioPlayer()
    player.play_audio("alphabet_audio/wav/t.wav")
    player.play_audio("alphabet_audio/wav/i.wav")
    player.play_audio("alphabet_audio/wav/m.wav")
    player.play_audio("alphabet_audio/wav/o.wav")
