from load_tester_utils import *
import subprocess
import datetime
import logging
import pytz
import threading

PST = pytz.timezone('US/Pacific')
now_pst = datetime.datetime.now(PST)

HOST_URL = "http://128.105.146.155/"

_R_values = [x for x in range(10,90,10)]
_R_values += [90,90,90,90,90,90,130,130,130,130]
# _R_values = [x for x in range(100,510,100)]


_current = 0
def next_value():
    global _current
    if _current >= len(_R_values):
        raise StopIteration("No more values")
    val = _R_values[_current]
    _current += 1
    return val

def get_traffic_rate(dt: datetime):
    rate = next_value()
    return max(int(rate), MIN_RATE)


def run_load_test():
    now = datetime.datetime.now()
    rate = get_traffic_rate(now)

    cmd = f"wrk2 -t{THREADS} -c{CONNECTIONS} -d{DURATION_SECONDS}s -R{rate} --latency {HOST_URL}"
    print_and_log(f"Test Started: {now.hour}:{now.minute} rate: {rate} pods: {get_hpa_info()[0]}")

    pod_samples = []
    sampler = threading.Thread(target=sample_pod_counts, args=(DURATION_SECONDS, POLL_INTERVAL, pod_samples))
    sampler.start()

    try:
        result = subprocess.run(cmd, shell=True, check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, timeout=DURATION_SECONDS+8)

        sampler.join()  # Wait for pod sampling to complete

        logging.info(f"WRK2 Output:\n{result.stdout}")
        if result.stderr:
            logging.error(f"WRK2 Errors:\n{result.stderr}")

        parsed = parse_wrk2_output(result.stdout)
        avg_pods = calculate_weighted_pod_average(pod_samples)

        write_historical_data(HISTORICAL_DATA_FILE, now, avg_pods, rate, THREADS, CONNECTIONS, DURATION_SECONDS, parsed)
        
        print_and_log(f"Test completed: {cmd}")
    except subprocess.CalledProcessError as e:
        print_and_log(f"Test failed: {e.stderr}")
    except Exception as e:
        print_and_log(f"Error: {str(e)}")



if __name__ == "__main__":
    initialize_historical_data()
    info1 = (f"Test Stats: \nPEAK_PARAMS = {PEAK_PARAMS}\nMAX_RATE = {MAX_RATE}\nMIN_RATE = {MIN_RATE}\nTHREADS = {THREADS}\nCONNECTIONS = {CONNECTIONS}")
    info2 = (f"Test start time: {timestamp_str}")
    print_and_log(info1)
    print_and_log(info2)
    while True:
        run_load_test()