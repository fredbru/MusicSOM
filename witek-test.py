
def findKickSync(low, mid, high, i, salienceProfile):
    # find instances  when kick syncopates against hi hat/snare on the beat. looking for kick proceeded by another hit
    # on a weaker metrical position
    kickSync = 0
    nextHit = ""
    if low[i] == 1.0 and low[(i+1)%32] != 1.0:
        for j in range(i+1, i+len(low)):
            if low[(j%32)] == 1.0:
                nextHit = "Kick"
                break
            elif mid[(j%32)] == 1.0 and high[(j%32)] != 1.0:
                nextHit = "Mid"
                break
            elif high[(j%32)] == 1.0 and mid[(j%32)] != 1.0:
                nextHit = "High"
                break
            elif high[(j%32)] == 1.0 and mid[(j%32)] == 1.0:
                nextHit = "MidAndHigh"
                break

        if nextHit == "MidAndHigh":
            if salienceProfile[j%32] >= salienceProfile[i]: #if hi hat is on a stronger beat - syncopation
                difference = salienceProfile[i] - salienceProfile[j%32]
                kickSync = difference + 2
                print("mid high sync")
        elif nextHit == "Mid":
            if salienceProfile[j%32] >= salienceProfile[i]: #if hi hat is on a stronger beat - syncopation
                difference = salienceProfile[i] - salienceProfile[j%32]
                kickSync = difference + 2
                print("mid sync")
        elif nextHit == "High":
            if salienceProfile[j%32] >= salienceProfile[i]:
                difference = salienceProfile[i] - salienceProfile[j%32]
                kickSync = difference + 5
                print("high sync")
    return kickSync

def findSnareSync(low, mid, high, i, salienceProfile):
    # find instances  when snare syncopates against hi hat/kick on the beat
    # S = n - ndi + I
    snareSync = 0
    nextHit = ""
    if mid[i] == 1.0 and mid[(i+1)%32] != 1.0:
        print(i)
        for j in range(i+1, i+len(mid)):
            if mid[(j%32)] == 1.0:
                nextHit = "Mid"
                break
            elif low[(j%32)] == 1.0 and high[(j%32)] != 1.0:
                nextHit = "Low"
                break
            elif high[(j%32)] == 1.0 and low[(j%32)] != 1.0:
                nextHit = "High"
                break
            elif high[(j%32)] == 1.0 and low[(j%32)] == 1.0:
                nextHit = "LowAndHigh"
                break
        if nextHit == "LowAndHigh":
            if salienceProfile[j%32] >= salienceProfile[i]:
                difference = salienceProfile[i] - salienceProfile[j%32]
                snareSync = difference + 1
                print("low high sync")
        elif nextHit == "Low":
            if salienceProfile[j%32] >= salienceProfile[i]:
                difference = salienceProfile[i] - salienceProfile[j%32]
                snareSync = difference + 1
                print("low sync")
        elif nextHit == "High":
            if salienceProfile[j%32] >= salienceProfile[i]: #if hi hat is on a stronger beat - syncopation
                difference = salienceProfile[i] - salienceProfile[j%32]
                snareSync = difference + 5
                print("high sync snare")
    return snareSync

def findHiHatSync(low, mid, high, i, salienceProfile):
    # find instances  when hiaht syncopates against snare/kick on the beat. this is my own adaptation of Witek 2014
    # may or may not work. currently doesn't consider velocity or open hi hats
    hihatSync = 0
    if high[i] == 1.0:
        if low[(i+1)%32] == 1.0:
            if salienceProfile[i+1] > salienceProfile[i]:
                hihatSync = 1 ### bit of a guess - maybe should be 0.5?
        elif mid[(i+1)%32] == 1.0:
            if salienceProfile[i + 1] > salienceProfile[i]:
                hihatSync =1 ### another guess
    return hihatSync

# Make a few test rhythms from Witek paper examples (figure S1)
hihatRhythm = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,
               1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,]

snareRhythm1 = [0,0,0,0,0,0,1,1,1,0,0,1,1,0,0,0,
               0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0]
kickRhythm1  = [1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,1,
               1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1]
s1 = 0

snareRhythm3 = [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,
                0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0]
kickRhythm3  = [1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,
               1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0]
s3 = 2

snareRhythm7 = [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,
                0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0]
kickRhythm7 =  [1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,
               1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
s7 = 7

snareRhythm8 = [0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,
                0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0]
kickRhythm8 = [1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,
               1,1,1,0,1,0,1,0,0,0,0,0,0,0,1,0]
s8 = 8

snareRhythm10 = [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,
                 0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0]
kickRhythm10 = [1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,
                1,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0]
s10 = 13

snareRhythm13 = [0,0,0,0,1,0,0,1,0,0,1,0,1,0,0,0,
                 0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]
kickRhythm13  = [1,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,
                0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0]
s13 = 17

snareRhythm15 = [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,
                 0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]
kickRhythm15  = [1,0,0,0,0,0,0,1,1,0,1,0,0,1,0,1,
                1,0,1,0,0,1,1,0,1,0,0,0,0,0,0,0]
s15 = 19

def testSync(kick,snare, hihat):
    salienceProfile = [0, -3, -2, -3, -1, -3, -2, -3, -1, -3, -2, -3, -1, -3, -2, -3,
                       0, -3, -2, -3, -1, -3, -2, -3, -1, -3, -2, -3, -1, -3, -2, -3]
    totalSyncopation = 0
    for i in range(len(hihatRhythm)):
        print(i)
        kickSync = findKickSync(kick, snare, hihat, i, salienceProfile)
        snareSync = findSnareSync(kick, snare, hihat, i, salienceProfile)
        totalSyncopation += kickSync
        totalSyncopation += snareSync
    return totalSyncopation


# Get syncopation distance for loop based on Witek 2014
# salience profile different to gomez/longuet higgins + lee
# Combines insrument weighting for cross-instrument syncopation. For now - just considering witek's 3 syncopation
# types with 3 kit parts. Later look at implementing 4 parts: add open hi hat, and syncopation with hi hat off pulse
# then look at adding velocity somehow (use velocity of syncopating part?)


sync1 = testSync(kickRhythm1,snareRhythm1,hihatRhythm)
print(sync1, s1)

sync3 = testSync(kickRhythm3,snareRhythm3,hihatRhythm)
print(sync3, s3)

sync7 = testSync(kickRhythm7,snareRhythm7,hihatRhythm)
print(sync7, s7)

sync8 = testSync(kickRhythm8,snareRhythm8,hihatRhythm)
print(sync8, s8)

sync10 = testSync(kickRhythm10,snareRhythm10,hihatRhythm)
print(sync10, s10)

sync13 = testSync(kickRhythm13,snareRhythm13,hihatRhythm)
print(sync13, s13)

sync15 = testSync(kickRhythm15,snareRhythm15,hihatRhythm)
print(sync15, s15)
