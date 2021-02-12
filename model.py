# Main Python file for Image detection
from car_detection import train_model, predict

# Sys & getopt for commandline Args
import sys
import getopt


def usage():
    usg = """
Prediction: 
python model.py -i <image file name> predict

Training model:
python model.py train

Help:
python model.py help
"""
    print(usg)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage()
        raise Exception('Error occured, cannot proceed further')
    else:
        opt, arg = getopt.getopt(sys.argv[1:], 'i:', ['image-file='])
        # print(opt)
        # print(arg)

        # As per arguments, follow action
        if len(arg) != 1:
            usage()
            raise Exception('Error occured, cannot proceed further')

        elif arg[0] == 'train':
            train_model()

        elif arg[0] == 'predict':
            if len(opt) > 0:
                opt = dict(opt)

                # Check file name option
                if '-i' in opt.keys() or '--image-file' in opt.keys():
                    img_file = opt['-i'] if '-i' in opt.keys() else opt['--image-file']

                    print(predict(img_file))

                else:
                    usage()
                    raise Exception('Error occured, cannot proceed further')

            else:
                usage()
                raise Exception('Error occured, cannot proceed further')

        elif arg[0] == 'help':
            usage()

        else:
            usage()
            raise Exception('Error occured, cannot proceed further')
