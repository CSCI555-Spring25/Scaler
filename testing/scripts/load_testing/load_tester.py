from load_tester_utils import *
import subprocess
import datetime
import logging
import pytz
import threading
import traceback

PST = pytz.timezone('US/Pacific')
now_pst = datetime.datetime.now(PST)

HOST_URL = "http://130.127.132.204/"

def run_load_test():
    now = datetime.datetime.now()
    rate = calculate_traffic_rate(now)

    init_pod_samples_csv()

    cmd = f"wrk2 -t{THREADS} -c{CONNECTIONS} -d{DURATION_SECONDS}s -R{rate} --latency {HOST_URL}"
    print_and_log(f"Test Started: {now.hour}:{now.minute} rate: {rate} pods: {get_hpa_info()[0]}")

    pod_samples = []
    sampler = threading.Thread(
        target=sample_pod_counts,
        args=(DURATION_SECONDS, POLL_INTERVAL, pod_samples, now)
    )
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
        print_and_log("An unexpected error occurred during load test.")
        print_and_log(f"Exception: {repr(e)}")
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        print_and_log(f"Full traceback:\n{tb}")



if __name__ == "__main__":
    initialize_historical_data()
    info1 = (f"Test Stats: \nPEAK_PARAMS = {PEAK_PARAMS}\nMAX_RATE = {MAX_RATE}\nMIN_RATE = {MIN_RATE}\nTHREADS = {THREADS}\nCONNECTIONS = {CONNECTIONS}")
    info2 = (f"Test start time: {timestamp_str}")
    print_and_log(info1)
    print_and_log(info2)
    while True:
        run_load_test()