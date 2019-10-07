import sys, json, random

def main():

    data = dict()
    with open(sys.argv[1], "r") as inFile:
        data = json.load(inFile)
    inFile.close()

    evalData = dict()

    for artist in data.keys():
        evalSongs = []
        range_ub = 35
        for i in range(10):
            try:
                randnum = random.randint(0, range_ub)
                evalSongs.append(data[artist].pop(randnum))
                range_ub -= 1
            except IndexError:
                print(randnum)
                exit(0)
        
        evalData[artist] = evalSongs.copy()

    with open(sys.argv[2], "w") as testFile:
        json.dump(data, testFile, indent=4)
    testFile.close()

    with open(sys.argv[3], "w") as evalFile:
        json.dump(evalData, evalFile, indent=4)
    evalFile.close()

    return 0

if __name__=="__main__":
    main()