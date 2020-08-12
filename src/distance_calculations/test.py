import math
cor1=[30.673855, 30.068294]
cor2=[30.674033, 30.068556]


def calc_dist_two_coor(current_coor,location):
    # 0 -> latitude
    # 1 -> longitude
    R = 6373.0
    i = 0
    current_coor = {'lon': math.radians(current_coor[1]),
                    'lat': math.radians(current_coor[0])}
    location = {'lon': math.radians(location[1]),
                'lat': math.radians(location[0])}
    dlon = location['lon'] - current_coor['lon']
    dlat = location['lat'] - current_coor['lat']
    a = math.sin(dlat / 2)**2 + math.cos(current_coor['lat']) * math.cos(
        location['lat']) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    meter = (int)(distance * 1000.0)
    return meter
print(calc_dist_two_coor(cor1,cor2))