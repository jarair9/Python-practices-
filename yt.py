from pytubefix import YouTube




def downloader(url):
    try:
        yt = YouTube(url, use_oauth=False, allow_oauth_cache=True)

        stream = (
            yt.streams
            .filter(progressive=True, file_extension='mp4')
            .order_by('resolution')
            .first()
        )

        if stream is None:
            print("No downloadable stream found.")
            return False

        stream.download(filename="video.mp4")
        print("The video downloaded successfully!")
        return True

    except Exception as e:
        print("An error occurred:", e)
        return False
url = str(input("Enter url: "))
downloader(url)