"""This module provides the transcript cli."""
import json
import shutil
import subprocess
from clint.textui import progress
import pytube
from moviepy.editor import VideoFileClip
import whisper
import os
import static_ffmpeg
from app import __version__
import requests
import re
from urllib.parse import urlparse, parse_qs
import time
from dotenv import dotenv_values
import yt_dlp
from deepgram import Deepgram
import mimetypes


def download_video(url):
    try:
        print(f"URL: {url}\nDownloading video... Please wait.")
        ydl_opts = {
            'format': '18',
            'outtmpl': 'tmp/videoFile.%(ext)s',
            'nopart': True,
            'writeinfojson': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ytdl:
            ytdl.download([url])

        with open('tmp/videoFile.info.json') as file:
            info = ytdl.sanitize_info(json.load(file))
            name = info['title'].replace('/', '-')
            file.close()

        os.rename("tmp/videoFile.mp4", f"tmp/{name}.mp4")

        return os.path.abspath(f"tmp/{name}.mp4")
    except Exception as e:
        print("Error downloading video")
        shutil.rmtree('tmp')
        return None


def read_description(prefix):
    try:
        list_of_chapters = []
        with open(prefix + 'videoFile.info.json', 'r') as f:
            info = json.load(f)
        if 'chapters' not in info:
            print("No chapters found in description")
            return list_of_chapters
        for index, x in enumerate(info['chapters']):
            name = x['title']
            start = x['start_time']
            list_of_chapters.append((str(index), start, str(name)))

        return list_of_chapters
    except Exception as e:
        print("Error reading description")
        return []


def write_chapters_file(chapter_file: str, chapter_list: list) -> None:
    try:
        with open(chapter_file, 'w') as fo:
            for current_chapter in chapter_list:
                fo.write(f'CHAPTER{current_chapter[0]}='
                         f'{current_chapter[1]}\n'
                         f'CHAPTER{current_chapter[0]}NAME='
                         f'{current_chapter[2]}\n')
    except Exception as e:
        print("Error writing chapter file")
        print(e)


def convert_video_to_mp3(filename):
    try:
        clip = VideoFileClip(filename)
        print(f"Converting video to mp3... Please wait.\n{filename[:-4]}.mp3")
        clip.audio.write_audiofile(filename[:-4] + ".mp3")
        clip.close()
        print("Converted video to mp3")
    except:
        print("Error converting video to mp3")
        return None
    return filename


def convert_wav_to_mp3(abs_path, filename):
    subprocess.call(['ffmpeg', '-i', abs_path, abs_path[:-4] + ".mp3"])
    return filename[:-4] + ".mp3"


def check_if_playlist(media):
    try:
        if media.startswith(("PL", "UU", "FL", "RD")):
            return True
        playlists = list(pytube.Playlist(media).video_urls)
        return type(playlists) is list
    except:
        return False


def check_if_video(media):
    if re.search(r'^([\dA-Za-z_-]{11})$', media):
        return True
    try:
        pytube.YouTube(media)
        return True
    except:
        return False


def get_playlist_videos(url):
    try:
        return pytube.Playlist(url)
    except Exception as e:
        print("Error getting playlist videos")
        print(e)
        return None


def get_audio_file(url, title):
    print("URL: " + url)
    print("downloading audio file")
    try:
        audio = requests.get(url, stream=True)
        with open("tmp/" + title + ".mp3", "wb") as f:
            total_length = int(audio.headers.get('content-length'))
            for chunk in progress.bar(audio.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
                if chunk:
                    f.write(chunk)
                    f.flush()
        return title + ".mp3"
    except Exception as e:
        print("Error downloading audio file")
        print(e)
        return None


def process_mp3(filename, model):
    print("Transcribing audio to text using whisper ...")
    try:
        my_model = whisper.load_model(model)
        result = my_model.transcribe(filename)
        data = [(x["start"], x["end"], x["text"]) for x in result["segments"]]
        print("Removed video and audio files")
        return data
    except Exception as e:
        print("Error transcribing audio to text")
        print(e)
        return None


def decimal_to_sexagesimal(dec):
    sec = int(dec % 60)
    minu = int((dec // 60) % 60)
    hrs = int((dec // 60) // 60)

    return f'{hrs:02d}:{minu:02d}:{sec:02d}'


def combine_chapter(chapters, transcript):
    try:
        chapters_pointer = 0
        transcript_pointer = 0
        result = ""
        # chapters index, start time, name
        # transcript start time, end time, text

        while chapters_pointer < len(chapters) and transcript_pointer < len(transcript):
            if chapters[chapters_pointer][1] <= transcript[transcript_pointer][0]:
                result = append_chapter_heading(result, chapters[chapters_pointer][2])
                chapters_pointer += 1
            else:
                result = append_transcript_text(result, transcript[transcript_pointer][2])
                transcript_pointer += 1

        while transcript_pointer < len(transcript):
            result = append_transcript_text(result, transcript[transcript_pointer][2])
            transcript_pointer += 1

        write_result_to_file(result, "result.md")
        return result
    except Exception as e:
        print("Error combining chapters")
        print(e)


def append_chapter_heading(result, chapter_name):
    result += f"\n\n## {chapter_name}\n\n"
    return result


def append_transcript_text(result, transcript_text):
    result += transcript_text
    return result


def write_result_to_file(result, filename):
    with open(filename, "w") as file:
        file.write(result)


def combine_deepgram_chapters_with_diarization(deepgram_data, chapters):
    try:
        para = ""
        string = ""
        curr_speaker = None
        words = deepgram_data["results"]["channels"][0]["alternatives"][0]["words"]
        words_pointer = 0
        chapters_pointer = 0

        while chapters_pointer < len(chapters) and words_pointer < len(words):
            if chapters[chapters_pointer][1] <= words[words_pointer]["start"]:
                string, para = append_chapter_heading(string, para, chapters[chapters_pointer][2])
                chapters_pointer += 1
            else:
                string, para, curr_speaker = process_word(string, para, curr_speaker, words[words_pointer])

            words_pointer += 1

        while words_pointer < len(words):
            string, para, curr_speaker = process_word(string, para, curr_speaker, words[words_pointer])
            words_pointer += 1

        para = para.strip(" ")
        string = string + para
        return string
    except Exception as e:
        print("Error combining deepgram chapters")
        print(e)


def append_chapter_heading(string, para, chapter_name):
    if para != "":
        para = para.strip(" ")
        string = string + para + "\n\n"
        para = ""
    string = string + f'## {chapter_name}\n\n'
    return string, para


def process_word(string, para, curr_speaker, word):
    if word["speaker"] != curr_speaker:
        if para != "":
            para = para.strip(" ")
            string = string + para + "\n\n"
            para = ""
        string = string + f'Speaker {word["speaker"]}: {decimal_to_sexagesimal(word["start"])}\n\n'
        curr_speaker = word["speaker"]

    para = para + " " + word["punctuated_word"]
    return string, para, curr_speaker


def get_deepgram_transcript(deepgram_data, diarize):
    if diarize:
        return generate_transcript_with_diarization(deepgram_data)
    else:
        return deepgram_data["results"]["channels"][0]["alternatives"][0]["transcript"]


def generate_transcript_with_diarization(deepgram_data):
    para = ""
    string = ""
    curr_speaker = None
    for word in deepgram_data["results"]["channels"][0]["alternatives"][0]["words"]:
        if word["speaker"] != curr_speaker:
            string, para = update_string_and_para(string, para, word)
            curr_speaker = word["speaker"]
        para += " " + word["punctuated_word"]
    string += para.strip(" ")
    return string


def update_string_and_para(string, para, word):
    if para != "":
        string += para.strip(" ") + "\n\n"
    para = ""
    string += f'Speaker {word["speaker"]}: {decimal_to_sexagesimal(word["start"])}\n\n'
    return string, para


def get_deepgram_summary(deepgram_data):
    try:
        summaries = deepgram_data["results"]["channels"][0]["alternatives"][0]["summaries"]
        return " ".join(x["summary"] for x in summaries).strip(" ")
    except Exception as e:
        print("Error getting summary")
        print(e)


def get_deepgram_transcript(deepgram_data, diarize):
    if diarize:
        return generate_transcript_with_diarization(deepgram_data)
    else:
        return deepgram_data["results"]["channels"][0]["alternatives"][0]["transcript"]


def generate_transcript_with_diarization(deepgram_data):
    para = ""
    string = ""
    curr_speaker = None
    for word in deepgram_data["results"]["channels"][0]["alternatives"][0]["words"]:
        if word["speaker"] != curr_speaker:
            string, para = update_string_and_para(string, para, word)
            curr_speaker = word["speaker"]
        para += " " + word["punctuated_word"]
    string += para.strip(" ")
    return string


def update_string_and_para(string, para, word):
    if para != "":
        string += para.strip(" ") + "\n\n"
    para = ""
    string += f'Speaker {word["speaker"]}: {decimal_to_sexagesimal(word["start"])}\n\n'
    return string, para


def get_deepgram_summary(deepgram_data):
    try:
        summaries = deepgram_data["results"]["channels"][0]["alternatives"][0]["summaries"]
        return " ".join(x["summary"] for x in summaries).strip(" ")
    except Exception as e:
        print("Error getting summary")
        print(e)


def create_transcript(data):
    return " ".join(x[2] for x in data)


def initialize():
    print('''
    This tool will convert Youtube videos to mp3 files and then transcribe them to text.
    ''')
    # FFMPEG installed on first use.
    print("Initializing FFMPEG...")
    static_ffmpeg.add_paths()
    print("Initialized FFMPEG")


def write_to_file(result, loc, url, title, date, tags, category, speakers, video_title, username, local, test, pr,
                  summary):
    try:
        transcribed_text = result
        file_title = get_file_title(title, video_title)
        meta_data = create_meta_data(file_title, url, tags, category, speakers, username, local, date, summary)
        file_name_with_ext = create_markdown_file(file_title, video_title, transcribed_text, test, pr)

        if local:
            url = None
        if not pr:
            generate_payload(loc=loc, title=file_title, transcript=transcribed_text, media=url, tags=tags,
                             category=category, speakers=speakers, username=username, event_date=date, test=test)
        return file_name_with_ext
    except Exception as e:
        print("Error writing to file")
        print(e)


def get_file_title(title, video_title):
    return title if title else video_title


def create_meta_data(file_title, url, tags, category, speakers, username, local, date, summary):
    meta_data = '---\n' \
                f'title: {file_title}\n' \
                f'transcript_by: {username} via TBTBTC v{__version__}\n'
    meta_data += f'media: {url}\n' if not local else ''
    meta_data += format_meta_list('tags', tags)
    meta_data += format_meta_list('speakers', speakers)
    meta_data += format_meta_list('categories', category)
    meta_data += f'summary: {summary}\n' if summary else ''
    meta_data += f'date: {date}\n' if date else ''
    meta_data += '---\n'
    return meta_data


def format_meta_list(meta_name, meta_value):
    if meta_value:
        meta_value = meta_value.strip().split(",")
        meta_value = [item.strip() for item in meta_value]
        return f'{meta_name}: {meta_value}\n'
    return ''


def create_markdown_file(file_title, video_title, transcribed_text, test, pr):
    file_name = video_title.replace(' ', '-')
    file_name_with_ext = "tmp/" + file_name + '.md'

    if test is not None or pr:
        with open(file_name_with_ext, 'a') as opf:
            opf.write(meta_data + '\n')
            opf.write(transcribed_text + '\n')
            opf.close()
    return file_name_with_ext


def get_md_file_path(result, loc, video, title, event_date, tags, category, speakers, username, local, video_title,
                     test, pr, summary=""):
    print("writing .md file")
    file_name_with_ext = write_to_file(result, loc, video, title, event_date, tags, category, speakers, video_title,
                                       username, local, test, pr, summary)
    print("wrote .md file")

    absolute_path = os.path.abspath(file_name_with_ext)
    return absolute_path


def create_pr(absolute_path, loc, username, curr_time, title):
    branch_name = loc.replace("/", "-")
    subprocess.call(['bash', 'initializeRepo.sh', absolute_path, loc, branch_name, username, curr_time])
    subprocess.call(['bash', 'github.sh', branch_name, username, curr_time, title])
    print("Please check the PR for the transcription.")


def get_username():
    if os.path.isfile(".username"):
        with open(".username", "r") as f:
            username = f.read()
    else:
        print("What is your github username?")
        username = input()
        with open(".username", "w") as f:
            f.write(username)
    return username


def check_source_type(source):
    if source.endswith(".mp3") or source.endswith(".wav"):
        if os.path.isfile(source):
            return "audio-local"
        else:
            return "audio"
    elif check_if_playlist(source):
        return "playlist"
    elif os.path.isfile(source):
        return "video-local"
    elif check_if_video(source):
        return "video"
    else:
        return None


def process_audio(source, title, event_date, tags, category, speakers, loc, model, username, local,
                  created_files, test, pr, deepgram, summarize, diarize):
    try:
        print("audio file detected")
        curr_time = str(round(time.time() * 1000))

        # Validate title
        if title is None:
            print("Error: Please supply a title for the audio file")
            return None

        # Get file path and process audio
        filename, abs_path = get_audio_file_path_and_process(source, title, local, created_files)
        if filename is None:
            print("File not found")
            return

        # Convert wav to mp3 if necessary
        if filename.endswith('wav'):
            initialize()
            abs_path = convert_wav_to_mp3(abs_path=abs_path, filename=filename)
            created_files.append(abs_path)

        # Get transcript result
        result = get_audio_transcript(abs_path, deepgram, summarize, diarize, test, model)

        # Create markdown file
        absolute_path = create_markdown_file(result, source, title, event_date, tags, category, speakers, username, local, filename, test, pr, summary)

        # Add created files and create PR if necessary
        handle_created_files_and_pr(absolute_path, created_files, pr, loc, username, curr_time, title)

        return absolute_path
    except Exception as e:
        print("Error processing audio file")
        print(e)


def get_audio_file_path_and_process(source, title, local, created_files):
    if not local:
        filename = get_audio_file(url=source, title=title)
        abs_path = os.path.abspath(path="tmp/" + filename)
        print("filename", filename)
        print("abs_path", abs_path)
        created_files.append(abs_path)
    else:
        filename = source.split("/")[-1]
        abs_path = source
    print("processing audio file", abs_path)
    return filename, abs_path


def get_audio_transcript(abs_path, deepgram, summarize, diarize, test, model):
    summary = None
    result = None
    if test:
        result = test
    else:
        if deepgram or summarize:
            deepgram_resp = process_mp3_deepgram(filename=abs_path, summarize=summarize, diarize=diarize)
            result = get_deepgram_transcript(deepgram_data=deepgram_resp, diarize=diarize)
            if summarize:
                summary = get_deepgram_summary(deepgram_data=deepgram_resp)
        if not deepgram:
            result = process_mp3(abs_path, model)
            result = create_transcript(result)
    return result


def create_markdown_file(result, source, title, event_date, tags, category, speakers, username, local, filename, test, pr, summary):
    return get_md_file_path(result=result, loc=loc, video=source, title=title, event_date=event_date,
                            tags=tags, category=category, speakers=speakers, username=username,
                            local=local, video_title=filename[:-4], test=test, pr=pr, summary=summary)


def handle_created_files_and_pr(absolute_path, created_files, pr, loc, username, curr_time, title):
    created_files.append(absolute_path)
    if pr:
        create_pr(absolute_path=absolute_path, loc=loc, username=username, curr_time=curr_time, title=title)
    else:
        created_files.append(absolute_path)


def process_videos(source, title, event_date, tags, category, speakers, loc, model, username, created_files,
                   chapters, pr, deepgram, summarize, diarize):
    try:
        print("Playlist detected")
        playlist_id = extract_playlist_id(source)
        url = "https://www.youtube.com/playlist?list=" + playlist_id
        print(url)
        videos = get_playlist_videos(url)
        if videos is None:
            print("Playlist is empty")
            return

        selected_model = model + '.en'
        filename = ""

        for video in videos:
            filename = process_video_for_each_playlist_item(video, title, event_date, tags, category, speakers,
                                                            loc, selected_model, username, pr, created_files,
                                                            chapters, diarize, deepgram, summarize)
            if filename is None:
                return None
        return filename
    except Exception as e:
        print("Error processing playlist")
        print(e)


def extract_playlist_id(source):
    if source.startswith("http") or source.startswith("www"):
        parsed_url = urlparse(source)
        return parse_qs(parsed_url.query)["list"][0]
    return source


def process_video_for_each_playlist_item(video, title, event_date, tags, category, speakers, loc, selected_model,
                                         username, pr, created_files, chapters, diarize, deepgram, summarize):
    return process_video(video=video, title=title, event_date=event_date, tags=tags, category=category,
                         speakers=speakers, loc=loc, model=selected_model, username=username, pr=pr,
                         created_files=created_files, chapters=chapters, test=False, diarize=diarize,
                         deepgram=deepgram, summarize=summarize)


def combine_deepgram_with_chapters(deepgram_data, chapters):
    try:
        chapters_pointer = 0
        words_pointer = 0
        result = ""
        words = deepgram_data["results"]["channels"][0]["alternatives"][0]["words"]

        def append_chapter_heading():
            nonlocal result, chapters_pointer
            result += f"\n\n## {chapters[chapters_pointer][2]}\n\n"
            chapters_pointer += 1

        while chapters_pointer < len(chapters) and words_pointer < len(words):
            if chapters[chapters_pointer][1] <= words[words_pointer]["end"]:
                append_chapter_heading()
            else:
                result += words[words_pointer]["punctuated_word"] + " "
                words_pointer += 1

        # Append the final chapter heading and remaining content
        while chapters_pointer < len(chapters):
            append_chapter_heading()
        while words_pointer < len(words):
            result += words[words_pointer]["punctuated_word"] + " "
            words_pointer += 1

        return result
    except Exception as e:
        print("Error combining deepgram with chapters")
        print(e)


def process_video(video, title, event_date, tags, category, speakers, loc, model, username, created_files,
                  chapters, test, pr, local=False, deepgram=False, summarize=False, diarize=False):
    try:
        curr_time = str(round(time.time() * 1000))
        video, filename, abs_path = determine_video_path(video, local, created_files)
        if event_date is None:
            event_date = get_date(video)

        initialize()
        summary, result, deepgram_data = process_video_transcription(abs_path, deepgram, summarize, diarize, test, chapters)
        handle_chapters(abs_path, chapters, deepgram, deepgram_data, result, local, filename, created_files, test)
        title, result = prepare_title_and_result(title, test, deepgram, result, filename)

        print("Creating markdown file")
        absolute_path = generate_markdown(result, loc, video, title, event_date, tags, summary, category, speakers,
                                          username, filename, local, pr, test)
        created_files.append("tmp/" + filename[:-4] + '.description')
        if not test:
            if pr:
                create_pr(absolute_path, loc, username, curr_time, title)
            else:
                created_files.append(absolute_path)
        return absolute_path
    except Exception as e:
        print("Error processing video")
        print(e)


def determine_video_path(video, local, created_files):
    if not local:
        video = standardize_video_link(video)
        print("Transcribing video: " + video)
        abs_path = download_video(url=video)
        if abs_path is None:
            print("File not found")
            return None
        created_files.append(abs_path)
        filename = abs_path.split("/")[-1]
    else:
        filename = video.split("/")[-1]
        print("Transcribing video: " + filename)
        abs_path = video
    return video, filename, abs_path


def standardize_video_link(video):
    if "watch?v=" in video:
        parsed_url = urlparse(video)
        video = parse_qs(parsed_url.query)["v"][0]
    elif "youtu.be" in video or "embed" in video:
        video = video.split("/")[-1]
    return "https://www.youtube.com/watch?v=" + video


def process_video_transcription(abs_path, deepgram, summarize, diarize, test, chapters):
    summary = None
    result = ""
    deepgram_data = None
    if chapters and not test:
        chapters = read_description("tmp/")
    elif test:
        chapters = read_description("test/testAssets/")
    convert_video_to_mp3(abs_path[:-4] + '.mp4')
    if deepgram or summarize:
        deepgram_data, result, summary = process_audio_with_deepgram(abs_path, summarize, diarize)
    if not deepgram:
        result = process_mp3(abs_path[:-4] + ".mp3", model)
    return summary, result, deepgram_data


def process_audio_with_deepgram(abs_path, summarize, diarize):
    deepgram_data = process_mp3_deepgram(abs_path[:-4] + ".mp3", summarize=summarize, diarize=diarize)
    result = get_deepgram_transcript(deepgram_data=deepgram_data, diarize=diarize)
    summary = None
    if summarize:
        print("Summarizing")
        summary = get_deepgram_summary(deepgram_data=deepgram_data)
    return deepgram_data, result, summary


def handle_chapters(abs_path, chapters, deepgram, deepgram_data, result, local, filename, created_files, test):
    if chapters and len(chapters) > 0:
        print("Chapters detected")
        write_chapters_file(abs_path[:-4] + '.chapters', chapters)
        created_files.append(abs_path[:-4] + '.chapters')
        if deepgram:
            if diarize:
                result = combine_deepgram_chapters_with_diarization(deepgram_data=deepgram_data, chapters=chapters)
            else:
                result = combine_deepgram_with_chapters(deepgram_data=deepgram_data, chapters=chapters)
        else:
            result = combine_chapter(chapters=chapters, transcript=result)
        if not local:
            created_files.append(abs_path)
            created_files.append("tmp/" + filename[:-4] + '.chapters')
    else:
        if not test and not deepgram:
            result = create_transcript(result)
        elif not deepgram:
            result = ""
    return chapters


def prepare_title_and_result(title, test, deepgram, result, filename):
    if not title:
        title = filename[:-4]
    if not test and not deepgram:
        result = create_transcript(result)
    elif not deepgram:
        result = ""
    return title, result


def generate_markdown(result, loc, video, title, event_date, tags, summary, category, speakers, username, video_title, local, pr, test):
    return get_md_file_path(result=result, loc=loc, video=video, title=title, event_date=event_date,
                            tags=tags, summary=summary, category=category, speakers=speakers,
                            username=username, video_title=video_title, local=local, pr=pr, test=test)


def process_source(source, title, event_date, tags, category, speakers, loc, model, username, source_type,
                   created_files, chapters, local=False, test=None, pr=False, deepgram=False, summarize=False,
                   diarize=False):
    try:
        if not os.path.isdir("tmp"):
            os.mkdir("tmp")
        else:
            shutil.rmtree("tmp")
            os.mkdir("tmp")

        local_audio = source_type == 'audio-local'
        local_video = source_type == 'video-local'

        if source_type in ('audio', 'audio-local'):
            filename = process_audio(source=source, title=title, event_date=event_date, tags=tags, category=category,
                                     speakers=speakers, loc=loc, model=model, username=username, summarize=summarize,
                                     local=local_audio, created_files=created_files, test=test, pr=pr, deepgram=deepgram,
                                     diarize=diarize)
        elif source_type == 'playlist':
            filename = process_videos(source=source, title=title, event_date=event_date, tags=tags, category=category,
                                      speakers=speakers, loc=loc, model=model, username=username, summarize=summarize,
                                      created_files=created_files, chapters=chapters, pr=pr, deepgram=deepgram,
                                      diarize=diarize)
        else:
            filename = process_video(video=source, title=title, event_date=event_date, summarize=summarize,
                                     tags=tags, category=category, speakers=speakers, loc=loc, model=model,
                                     username=username, created_files=created_files, local=local or local_video, diarize=diarize,
                                     chapters=chapters, test=test, pr=pr, deepgram=deepgram)

        return filename
    except Exception as e:
        print("Error processing source")
        print(e)


def get_date(url):
    video = pytube.YouTube(url)
    return str(video.publish_date).split(" ")[0]


def clean_up(created_files):
    for file in created_files:
        if os.path.isfile(file):
            os.remove(file)
    shutil.rmtree("tmp")


def generate_payload(loc, title, event_date, tags, category, speakers, username, media, transcript, test):
    try:
        formatted_event_date = format_event_date(event_date)
        data = create_payload_data(title, category, tags, speakers, username, formatted_event_date, media, loc, transcript)
        content = {'content': data}

        if test:
            return content
        else:
            return post_payload_to_queue(content)
    except Exception as e:
        print(e)


def format_event_date(event_date):
    if event_date is None:
        return None
    if isinstance(event_date, str):
        return event_date
    return event_date.strftime('%Y-%m-%d')


def create_payload_data(title, category, tags, speakers, username, event_date, media, loc, transcript):
    return {
        "title": title,
        "transcript_by": f'{username} via TBTBTC v{__version__}',
        "categories": str(category),
        "tags": str(tags),
        "speakers": str(speakers),
        "date": event_date,
        "media": media,
        "loc": loc,
        "body": transcript
    }


def post_payload_to_queue(content):
    config = dotenv_values(".env")
    url = config['QUEUE_ENDPOINT'] + "/api/transcripts"
    resp = requests.post(url, json=content)

    if resp.status_code == 200:
        print("Transcript added to queue")

    return resp

