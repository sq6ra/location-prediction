import numpy as np
import math
import util

def read_loc_by_id(filename):
    lines = open(filename).readlines()
    loc_by_id = {}
    info_by_id = {}
    for line in lines[1:]:
        tokens = line.strip().split(",")
        pid = int(tokens[0])
        hours = int(tokens[1]), int(tokens[2]), int(tokens[3])
        lat = float(tokens[4])
        lng = float(tokens[5])
        posts = int(tokens[6])
        info_by_id[pid] = (hours, posts)
        if lat != 0 or lng != 0:
            loc_by_id[pid] = (lat, lng)
    return loc_by_id, info_by_id

def calculate_vector(pid, graph, df, info_by_id):
    vector = {}
    for friend in graph[pid]:
        vector[friend] = 1
    if pid in info_by_id:
        for hour in info_by_id[pid][0]:
            if hour != 25:
                vector[-hour] = 1.0 / 100
    return vector
    

def calculate_vector_for_query(pid, graph, df, info_by_id):
    vector = {pid : 2}
    for friend in graph[pid]:
        if friend in df:
##            vector[friend] = 1.0 / math.sqrt(df[friend])
            vector[friend] = 1.0 / math.sqrt(math.sqrt(df[friend]))
    if pid in info_by_id:
        for hour in info_by_id[pid][0]:
            if hour != 25:
                vector[-hour] = 1.0 / 20
    return vector

def make_invidx(graph, loc_by_id, df, info_by_id):
    feature_set = set(graph.keys())
    feature_set.update(range(0,-24,-1))

    invidx = {feature : {} for feature in feature_set}
    length_by_id = {}
    for pid in loc_by_id:
        if pid not in graph:
            continue
        vector = calculate_vector(pid, graph, df, info_by_id)
        length = math.sqrt(sum([value**2 for value in vector.values()]))
        length_by_id[pid] = length
        for feature in vector:
            invidx[feature][pid] = vector[feature] / length

    avg_length = sum(length_by_id.values()) / len(length_by_id)
    return invidx, length_by_id, avg_length

def do_query(paras, query, self_id, s):
    invidx, length_by_id, avg_length = paras
    simi = {}
    for feature in query:
        for pid in invidx.get(feature, {}):
            simi[pid] = simi.get(pid, 0) + query[feature]*invidx[feature][pid]
    return sorted([ \
        (simi[pid]/(avg_length/length_by_id[pid]*(1-s) + s), pid) \
        for pid in simi if pid != self_id], reverse=True)

def w_avg_top_k(pid):
    vector = calculate_vector_for_query(pid, graph, df)
    rank_list = do_query(paras, vector, pid, s)
    sum_lat = 0
    sum_lng = 0
    count = 0.0
    for (simi, pid) in rank_list[:k]:
        loc = loc_by_id[pid]
        print "%d\t%d\t%f\t%f\t%f" % (pid, len(graph[pid]), loc[0], loc[1], simi)
        count += simi
        sum_lat += loc[0] * simi
        sum_lng += loc[1] * simi
    print "avg: lat: %lf\tlng: %lf" % (sum_lat/count, sum_lng/count)

def avg_avg_predict(pid):
    lat_sum = 0
    lng_sum = 0
    count = 0.0
    for friend in graph[pid]:
        friend_lat_sum = 0
        friend_lng_sum = 0
        friend_count = 0.0
        for ff in graph[friend]:
            if ff == pid:
                continue
            if ff in loc_by_id:
                lat, lng = loc_by_id[ff]
                print "%d\t%d\t%d" % (ff, lat, lng)
                friend_lat_sum += lat
                friend_lng_sum += lng
                friend_count += 1
        if friend_count != 0:
            print "avg\t%d\t%f\t%f" % (friend, friend_lat_sum / friend_count, friend_lng_sum / friend_count)
            lat_sum += friend_lat_sum / friend_count
            lng_sum += friend_lng_sum / friend_count
            count += 1
    if count != 0:
        print lat_sum / count, lng_sum / count
    else:
        print "cannot predict"
    
def w_avg_top_k_predict(rank_list, loc_by_id, k):
    rank_list = rank_list[:k]
    if len(rank_list) == 0:
        return None
    sum_lat = 0
    sum_lng = 0
    weights = 0
    for (simi, pid) in rank_list:
        loc = loc_by_id[pid]
        weights += simi
        sum_lat += loc[0] * simi
        sum_lng += loc[1] * simi
    return sum_lat/weights, sum_lng/weights

def w_avg_top_k_predict_2(rank_list, loc_by_id, k):
    rank_list = rank_list[:k]
    if len(rank_list) == 0:
        return None
    sum_lat = 0
    sum_lng = 0
    weights = 0
    for (simi, pid) in rank_list:
        loc = loc_by_id[pid]
        weights += simi
        sum_lat += loc[0] * simi
        sum_lng += loc[1] * simi
    avg_lat, avg_lng = sum_lat/weights, sum_lng/weights
    if len(rank_list) < k:
        return avg_lat, avg_lng
    dist_list = []
    for (simi, pid) in rank_list:
        lat, lng = loc_by_id[pid]
        dist = (lat-avg_lat)**2+(lng-avg_lng)**2
        dist_list.append((dist, pid))
    dist, farthest_pid = max(dist_list)
    sum_lat = 0
    sum_lng = 0
    weights = 0
    for (simi, pid) in rank_list:
        if pid == farthest_pid:
            continue
        loc = loc_by_id[pid]
        weights += simi
        sum_lat += loc[0] * simi
        sum_lng += loc[1] * simi
    return sum_lat/weights, sum_lng/weights

def calculate_df(graph, loc_by_id, info_by_id):
    count = {}
    for pid in graph:
        if pid in loc_by_id:
            h1, h2, h3 = info_by_id[pid][0]
            if h1 != 25:
                count[-h1] = count.get(-h1, 0) + 1
            for friend in graph[pid]:
                count[friend] = count.get(friend, 0) + 1
    return count

def predict_all(graph, loc_by_id, info_by_id, paras, s, k, df, id_list):
    none_count = 0
    total_count = 0
    ans_loc_by_id = {}
    for pid in id_list:
        vector = calculate_vector_for_query(pid, graph, df, info_by_id)
        rank_list = do_query(paras, vector, pid, s)        
        if len(graph[pid]) <= 2:
            ans_loc = w_avg_top_k_predict(rank_list, loc_by_id, k)
        else:
            ans_loc = w_avg_top_k_predict_2(rank_list, loc_by_id, k)
        if ans_loc == None:
            none_count += 1
        else:
            ans_loc_by_id[pid] = ans_loc
        total_count += 1
        if total_count % 100 == 0:
            print total_count
    print "none_count:", none_count
    return ans_loc_by_id

def predict_one(pid, graph, loc_by_id, info_by_id, paras, s, k, df):
    vector = calculate_vector_for_query(pid, graph, df, info_by_id)
    rank_list = do_query(paras, vector, pid, s)
#    if len(graph[pid]) <= 3:
#        return w_avg_top_k_predict(rank_list, loc_by_id, k)
#    else:
    return w_avg_top_k_predict_2(rank_list, loc_by_id, k)

def parallel_predict_all(graph, loc_by_id, info_by_id, paras, s, k, df):
    from multiprocessing import Pool
    from functools import partial
    pid_list = [pid for pid in graph]
##    pid_list = [pid for pid in graph]
##    pid_list = [pid for pid in graph if len(graph[pid])==1 and pid in loc_by_id]
    p = Pool()
    partial_predict_one = partial(predict_one, graph=graph, \
                                  loc_by_id=loc_by_id, \
                                  info_by_id=info_by_id, paras=paras, s=s, k=k, df=df)
    ans_loc_list = p.map(partial_predict_one, pid_list)
##    ans_loc_list = map(partial_predict_one, pid_list)
    ans_loc_by_id = {}
    none_count = 0
    for (index, pid) in enumerate(pid_list):
        ans_loc = ans_loc_list[index]
        if ans_loc == None:
            none_count += 1
        else:
            ans_loc_by_id[pid] = ans_loc
    print "none_count:", none_count
    return ans_loc_by_id

def convert_to_ndarray(loc_by_id):
    new_loc_by_id = {pid : np.zeros(3, dtype=float) for pid in loc_by_id}
    for pid in loc_by_id:
        new_loc = new_loc_by_id[pid]
        loc = loc_by_id[pid]
        new_loc[0] = loc[0]
        new_loc[1] = loc[1]
        new_loc[2] = 1
    return new_loc_by_id

def evaluate(graph, loc_by_id, ans_loc_by_id):
    loc_by_id = convert_to_ndarray(loc_by_id)
    ans_loc_by_id = convert_to_ndarray(ans_loc_by_id)
    e_by_n = {}
    n_by_lat_e = {}
    n_by_lng_e = {}
    for pid in loc_by_id:
        if pid not in ans_loc_by_id:
            continue
        e = (loc_by_id[pid] - ans_loc_by_id[pid])**2
        e[2] = 1
        n = len(graph[pid])
        if n in e_by_n:
            e_by_n[n] += e
        else:
            e_by_n[n] = e
        lat_e = int(abs(loc_by_id[pid][0] - ans_loc_by_id[pid][0]))
        lng_e = int(abs(loc_by_id[pid][1] - ans_loc_by_id[pid][1]))
        n_by_lat_e[lat_e] = n_by_lat_e.get(lat_e, 0) + 1
        n_by_lng_e[lng_e] = n_by_lng_e.get(lng_e, 0) + 1        
    total_e = sum(e_by_n.values())
    total_e /= total_e[2]
    print "lat: %lf\tlng: %lf\ttotal: %lf" % (math.sqrt(total_e[0]), math.sqrt(total_e[1]), math.sqrt(total_e[0]+total_e[1]))
    for n in sorted(e_by_n.keys()):
        e = e_by_n[n]
        e /= e[2]
        print "%d\t%lf\t%lf\t%lf" % (n, math.sqrt(e[0]), math.sqrt(e[1]), math.sqrt(e[0]+e[1]))

##    print "LAT ERROR DISTRIBUTION"
##    for lat_e in sorted(n_by_lat_e.keys()):
##        print "%d\t%d" % (lat_e, n_by_lat_e[lat_e])
##    print "LNG ERROR DISTRIBUTION"
##    for lng_e in sorted(n_by_lng_e.keys()):
##        print "%d\t%d" % (lng_e, n_by_lng_e[lng_e])

def output_result(filename, ans_loc_by_id, test_id_list):
    output = open(filename, "w")
    output.write("Id,Lat,Lon\n")
    id_list = sorted(test_id_list)
    for pid in id_list:
        lat, lng = ans_loc_by_id[pid]
        output.write("%d,%f,%f\n" % (pid, lat, lng))
    output.close()

if __name__ == "__main__":
    loc_by_id, info_by_id = read_loc_by_id("./data/posts-train.txt")
    graph = util.read_graph("./data/graph.txt")

    test_info_by_id = util.read_test_set("./data/posts-test-x.txt")
    info_by_id.update(test_info_by_id)
    test_id_list = test_info_by_id.keys()
    
    import time
    start = time.time()
    s = 0.7
    k = 50
#### exactly avg_avg
##    s = 0
##    k = 40000
        
    df = calculate_df(graph, loc_by_id, info_by_id)
    paras = make_invidx(graph, loc_by_id, df, info_by_id)
##    ans_loc_by_id = parallel_predict_all(graph, loc_by_id, info_by_id, paras, s, k, df)
##    evaluate(graph, loc_by_id, ans_loc_by_id)

    ans_loc_by_id = predict_all(graph, loc_by_id, info_by_id, paras, s, k, df, test_id_list)        
    output_result("./submit/knn_k40_s05_sqrt_ws2_filtOut_wh.txt", ans_loc_by_id, test_id_list)
    end = time.time()
    print end-start, "s"
