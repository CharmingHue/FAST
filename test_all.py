from utils import ResultFormat
import subprocess
import argparse
import json
import os
import logging


def generate_results_from_json(min_score, dataset, result_path):
    rf = ResultFormat(dataset, result_path)
    output_json = os.path.join(args.output_path, 'output.json')
    file = open(output_json, "r")
    fileJson = json.load(file)
    fileJson = json.loads(fileJson)
    for k, v in fileJson.items():
        bboxes, scores = [], []
        for bbox, score in zip(v["bboxes"], v["scores"]):
            if score > min_score:
                bboxes.append(bbox)
                scores.append(score)
        v["bboxes"] = bboxes
        v["scores"] = scores
        rf.write_result(k, v)


def ctw_eval(ep):
    run_cmd = 'python test.py %s %s/checkpoint_%dep.pth.tar --min-score %s' % (args.config, args.checkpoint, ep, args.min_score)
    if args.ema:
        run_cmd += " --ema"
    print(run_cmd)
    logging.info(run_cmd)
    run_cmd = cd_root_cmd + " && " + run_cmd
    print('[epoch: %d/%d]' % (ep, args.end_ep))
    logging.info('[epoch: %d/%d]' % (ep, args.end_ep))
    p = subprocess.Popen(run_cmd, shell=True, stderr=subprocess.STDOUT)
    p.wait()
    f_list = []
    result_path = os.path.join(args.output_path, 'submit_ctw')
    for i in range(80, 95 + 1):
        generate_results_from_json(min_score=i/100, dataset='CTW', result_path=result_path)
        eval_cmd = cd_root_cmd + " && " + 'cd eval && sh eval_ctw.sh %s' % ('../../' + result_path)
        p = subprocess.Popen(eval_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()
        f = 0
        line = ''
        for line in iter(p.stdout.readline, b''):
            line = str(line).replace('\\n', '').replace('\n', '').replace('b\'', '').replace('\'', '')
            if args.debug: print(line)
            if 'f:' in line:
                print("Min-score: %.2f |" % (i / 100), line)
                logging.info("Min-score: %.2f | %s" % (i / 100, line))
                f = float(line.split('f: ')[-1])
                f_list.append(f)
                global best_f, best_ep, best_line, best_min_score
                if f >= best_f:
                    best_f = f
                    best_ep = ep
                    best_line = line
                    best_min_score = i / 100
                break
        p.stdout.close()
    print('best f-measure %f at ep %d with min-score %.2f' % (best_f, best_ep, best_min_score))
    logging.info('best f-measure %f at ep %d with min-score %.2f' % (best_f, best_ep, best_min_score))
    print(best_line)
    logging.info(best_line)
    return max(f_list)


def ic15_eval(ep):
    run_cmd = 'python test.py %s %s/checkpoint_%dep.pth.tar --min-score 0' % (args.config, args.checkpoint, ep)
    if args.ema:
        run_cmd += " --ema"
    print(run_cmd)
    logging.info(run_cmd)
    run_cmd = cd_root_cmd + " && " + run_cmd
    print('[epoch: %d/%d]' % (ep, args.end_ep))
    logging.info('[epoch: %d/%d]' % (ep, args.end_ep))
    p = subprocess.Popen(run_cmd, shell=True, stderr=subprocess.STDOUT)
    p.wait()
    f_list = []
    result_path = os.path.join(args.output_path, 'submit_ic15.zip')

    for i in range(80, 90 + 1):
        generate_results_from_json(min_score=i/100, dataset='IC15', result_path=result_path)
        eval_cmd = cd_root_cmd + " && " + 'cd eval && sh eval_ic15.sh'
        p = subprocess.Popen(eval_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()
        f = 0
        line = ''
        for line in iter(p.stdout.readline, b''):
            line = str(line).replace('\\n', '').replace('\n', '').replace('b\'', '').replace('\'', '')
            if args.debug: print(line)
            if 'hmean' in line:
                print("Min-score: %.2f |" % (i / 100), line)
                logging.info("Min-score: %.2f | %s" % (i / 100, line))
                f = float(line.split(', ')[-2].split('\"hmean\": ')[-1])
                f_list.append(f)
                global best_f, best_ep, best_line, best_min_score
                if f >= best_f:
                    best_f = f
                    best_ep = ep
                    best_line = line
                    best_min_score = i / 100
                break
        p.stdout.close()
    print('best f-measure %f at ep %d with min-score %.2f' % (best_f, best_ep, best_min_score))
    logging.info('best f-measure %f at ep %d with min-score %.2f' % (best_f, best_ep, best_min_score))
    print(best_line)
    logging.info(best_line)
    return max(f_list)


def tt_eval(ep):
    run_cmd = 'python test.py %s %s/checkpoint_%dep.pth.tar --min-score 0 ' % (args.config, args.checkpoint, ep)
    if args.ema:
        run_cmd += " --ema"
    print(run_cmd)
    logging.info(run_cmd)
    run_cmd = cd_root_cmd + " && " + run_cmd
    print('[epoch: %d/%d]' % (ep, args.end_ep))
    logging.info('[epoch: %d/%d]' % (ep, args.end_ep))
    p = subprocess.Popen(run_cmd, shell=True, stderr=subprocess.STDOUT)
    p.wait()
    f_list = []
    result_path = os.path.join(args.output_path, 'submit_tt')
    
    for i in range(80, 90 + 1):
        generate_results_from_json(min_score=i/100, dataset='TT', result_path=result_path)
        eval_cmd = cd_root_cmd + " && " + 'cd eval && sh eval_tt.sh %s' % ('../../' + result_path)
        p = subprocess.Popen(eval_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()

        f = 0
        line = ''
        for line in iter(p.stdout.readline, b''):
            line = str(line).replace('\\n', '').replace('\n', '').replace('b\'', '').replace('\'', '')
            if args.debug: print(line)
            if 'Hmean:' in line:
                print("Min-score: %.2f |" % (i / 100), line)
                logging.info("Min-score: %.2f | %s" % (i / 100, line))
                f = float(line.split('Hmean:')[-1])
                f_list.append(f)
                global best_f, best_ep, best_line, best_min_score
                if f >= best_f:
                    best_f = f
                    best_ep = ep
                    best_line = line
                    best_min_score = i / 100
                break
        p.stdout.close()
    print('best f-measure %f at ep %d with min-score %.2f'%(best_f, best_ep, best_min_score))
    logging.info('best f-measure %f at ep %d with min-score %.2f'%(best_f, best_ep, best_min_score))
    print(best_line)
    logging.info(best_line)
    return max(f_list)


def msra_eval(ep):
    run_cmd = 'python test.py %s %s/checkpoint_%dep.pth.tar --min-score 0' % (args.config, args.checkpoint, ep)
    if args.ema:
        run_cmd += " --ema"
    print(run_cmd)
    logging.info(run_cmd)
    run_cmd = cd_root_cmd + " && " + run_cmd
    print('[epoch: %d/%d]' % (ep, args.end_ep))
    logging.info('[epoch: %d/%d]' % (ep, args.end_ep))
    p = subprocess.Popen(run_cmd, shell=True, stderr=subprocess.STDOUT)
    p.wait()
    f_list = []
    result_path = os.path.join(args.output_path, 'submit_msra')
    
    for i in range(80, 95 + 1):
        generate_results_from_json(min_score=i/100, dataset='MSRA', result_path=result_path)
        eval_cmd = cd_root_cmd + " && " + 'cd eval && sh eval_msra.sh'
        p = subprocess.Popen(eval_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        p.wait()
        f = 0
        line = ''
        for line in iter(p.stdout.readline, b''):
            line = str(line).replace('\\n', '').replace('\n', '').replace('b\'', '').replace('\'', '')
            if args.debug: print(line)
            if 'f:' in line:
                print("Min-score: %.2f |" % (i / 100), line)
                logging.info("Min-score: %.2f | %s" % (i / 100, line))
                f = float(line.split('f: ')[-1])
                f_list.append(f)
                global best_f, best_ep, best_line, best_min_score
                if f >= best_f:
                    best_f = f
                    best_ep = ep
                    best_line = line
                    best_min_score = i / 100
                break
        p.stdout.close()
    print('best f-measure %f at ep %d with min-score %.2f'%(best_f, best_ep, best_min_score))
    logging.info('best f-measure %f at ep %d with min-score %.2f'%(best_f, best_ep, best_min_score))
    print(best_line)
    logging.info(best_line)
    return max(f_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', nargs='?', type=str)
    parser.add_argument('checkpoint', nargs='?', type=str)
    parser.add_argument('--dataset', nargs='?', type=str, required=True,
                        choices=['ctw', 'ic15', 'tt', 'msra'])
    parser.add_argument('--start-ep', nargs='?', type=int, required=True)
    parser.add_argument('--end-ep', nargs='?', type=int, required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--min-score', nargs='?', type=float, default=0) 
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    cd_root_cmd = 'cd %s' % root

    processes = []
    best_f = 0
    best_ep = 0
    best_min_score = 0
    best_line = ''

    if args.dataset == 'ctw':
        eval_ = ctw_eval
    elif args.dataset == 'ic15':
        eval_ = ic15_eval
    elif args.dataset == 'tt':
        eval_ = tt_eval
    elif args.dataset == 'msra':
        eval_ = msra_eval
        
    best_f_list = []
    
    config_head = args.config.split('/')[-1].split('.')[0]
    args.__dict__['output_path'] = os.path.join('outputs', config_head)
    os.makedirs(args.output_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.output_path, 'test_all_%s.log' % (config_head)), 
                        level=logging.INFO,
                        format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
                        datefmt='%Y-%m-%d %A %H:%M:%S')
    
    for ep in range(args.start_ep, args.end_ep + 1, 1):
        best_f_list.append(eval_(ep))
    
