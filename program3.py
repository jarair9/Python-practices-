# import random


# guess = 0
# digit = random.choice((1,10))
# attempts = 0
# while True:
#     num = int(input('enter a number : '))
#     if num > digit:
#         print('Lower number please')
#         attempts += 1
#     elif num < digit:
#         print('Higher number please')
#         attempts += 1
#     else:
#         print(f'You have guessed {digit} in {attempts}')
#         break
from pytubefix import YouTube


def downloader(url):
    try:
        yt = YouTube(url, use_oauth=False, allow_oauth_cache=True)

        stream = yt.streams.filter(progressive=True, file_extension='mp4')\
                           .order_by('resolution')\
                           .first()
        if stream is None:
            print("No downloadable stream found.")
            return

        stream.download(filename="video.mp4")
        print("The video downloaded successfully!")

    except Exception as e:
        print("An error occurred:", e)

url = input("Enter URL: ")
downloader(url)
