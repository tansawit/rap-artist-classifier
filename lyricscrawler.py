import sys, requests, os, json, csv, itertools, xlwt
from xlwt import Workbook
from urllib.parse import urljoin
from bs4 import BeautifulSoup

base_url = "http://api.genius.com"
requestheaders = {
    "Authorization": "Bearer mvC2-Cnzx-iATii17nTkjOQ8IXjcCbhmVXg-Yp35uhw6Ye8tEpweyhaVIUMDAe98",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36"

}


def get_artist_details(artistName):
    """Return id for artist which will be used to scrape their songs from API."""
    global base_url, requestheaders

    artistURL = urljoin(base_url, "search")

    apiresponse = requests.get(artistURL, params={"q": artistName}, headers = requestheaders).json()

    return apiresponse


def get_artist_songs(artistid):
    """Returns top N songs of artist."""
    global base_url, requestheaders

    apiurl = urljoin(base_url, str("artists/"+str(artistid)+"/songs"))

    apiresponse = requests.get(apiurl, params={"id": artistid, "sort": "popularity", "per_page": 50}, headers = requestheaders).json()

    return apiresponse

def get_song_lyrics(url):
    """Returns lyrics for the song."""
    pagecontent = BeautifulSoup(requests.get(url).text, 'html.parser')
    for header in pagecontent('script'):
        header.extract()

    lyrics = None
    try:
        lyrics = pagecontent.find('div', class_="lyrics").get_text()
    except AttributeError:
        return "Not Available!"
    return lyrics

def get_song_info(id):
    """Returns song info."""
    global base_url, requestheaders

    apiurl = urljoin(base_url, str("songs/"+str(id)))

    apiresponse = requests.get(apiurl, params={"q": id}, headers = requestheaders).json()

    return apiresponse 

def main():
    """Driver function for the crawler."""
    if len(sys.argv) != 3:
        print("Usage: lyricscrawler.py [filename containing artist list] [outputfile.csv]")
        exit(1)
    
    artistdata = {}
    artistsongs = {}

    #print(os.getcwd())

    with open(sys.argv[1], "r") as artistFile:
        for line in artistFile.readlines():
            artistdata[line.replace('\n', '')] = {}
            artistsongs[line.replace('\n', '')] = {}
    artistFile.close()
    
    #artist data will have key as artist name and dictionary as value.
    #The value dictionary will have songname and song lyrics as key and value.
    #artistdata = {}
    for artist in artistdata.keys():
        try:
            apiresponse = get_artist_details(artist.replace('\n', ''))
        except requests.RequestException as e:
            print("Web Request exception. Is the atist name correct?\n\n\n")
            print(e)
            continue
        
        #Debugging
        #print(apiresponse["response"]["hits"][0]["result"]["primary_artist"]["id"])

        #We do not need to store whole artist details reponse
        #Just keeping it for now if we need any more details
        artistdata[artist] = apiresponse["response"]["hits"][0]["result"]["primary_artist"]

    #Now we have details about all artists
    #All we need to do is use GET /artists/:id/songs and get their top N songs
    artistNum = 0
    songNum = 0
    for artist in artistsongs.keys():
        try:
            apiresponse = get_artist_songs(artistdata[artist]["id"])
            artistsongs[artist] = []
        except requests.RequestException as e:
            print("Web Request exception. Is the atist name correct?\n\n\n")
            print(e)
            continue
        

        for song in apiresponse["response"]["songs"]:
            #Get lyrics for the songs
            #Get rid of redundant info and just push useful info
            currSong = {}
            currSong["title"] = song["title_with_featured"]
            currSong["id"] = song["id"]
            try:
                currSong["lyrics"] = get_song_lyrics(song["url"])
                songinfo = get_song_info(currSong["id"])

            except requests.RequestException as e:
                print("Web Request exception. Is the songname correct?\n\n\n")
                print(e)
                continue
            currSong["release_date"] = songinfo["response"]["song"]["release_date"]
            currSong["recording_location"] = songinfo["response"]["song"]["recording_location"]
            (artistsongs[artist]).append(currSong)    
            #SongNum is just to know that program is still running
            songNum+=1    
            print(songNum)
        artistNum += 1

    #Convert artistsongs to csv if needed or write it to file
    """with open(sys.argv[2], "w", encoding="utf-8") as csv_file:
        csv_writer  = csv.writer(csv_file, delimiter='\t')
        csv_writer.writerow(['Artist', 'Song', 'Lyrics'])
        for artist in artistsongs.keys():
            for song in artistsongs[artist].keys():
                #line = [artist, song, artistsongs[artist][song]]
                csv_writer.writerow( [artist, song, artistsongs[artist][song]])
    csv_file.close()"""

    #Write it to excel/*-
    #Doesnt support more thanm 3k chars in cell so lyrics cant be written in excel cell
    #This may be the reason for CSV not working as well
    """wb = Workbook()
    sheet1 = wb.add_sheet('sheet 1')
    rownum = 0

    sheet1.write(0, rownum, 'Artist')
    sheet1.write(1, rownum, 'Song')
    sheet1.write(2, rownum, 'Lyrics')
    rownum += 1

    for artist in artistsongs.keys():
        for song in artistsongs[artist].keys():
            sheet1.write(0, rownum, artist)
            sheet1.write(1, row num, song)
            sheet1.write(2, rownum, artistsongs[artist][song])
            rownum += 1
    wb.save(sys.argv[2])"""

    #Just write JSON to output file
    #loaded_json = json.loads(artistsongs)
    with open(sys.argv[2], "w", encoding="utf-8") as outputfile:
        json.dump(artistsongs, outputfile, indent=4)
    outputfile.close()

    return 0

if __name__=="__main__":
    main()
