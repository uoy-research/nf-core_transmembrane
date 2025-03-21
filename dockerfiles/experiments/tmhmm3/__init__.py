"""
This file is part of the DeepTMHMM project.

For license information, please see the README.txt file in the root directory.
"""

import os
import pickle
import hashlib
import json
from deeptmhmm.util import load_model_from_disk, set_experiment_id, write_prediction_data_to_disk
from .tm_models import *
from .tm_util import *

def run_experiment(parser, use_gpu):
    parser.add_argument('--minibatch-size-validation',
                        dest='minibatch_size_validation',
                        type=int,
                        default=8,
                        help='Size of each minibatch during evaluation.')
    parser.add_argument('--hidden-size',
                        dest='hidden_size',
                        type=int,
                        default=64,
                        help='Hidden size.')
    parser.add_argument('--learning-rate',
                        dest='learning_rate',
                        type=float,
                        default=0.0002,
                        help='Learning rate to use during training.')
    parser.add_argument('--cv-partition',
                        dest='cv_partition',
                        type=int,
                        default=0,
                        help='Run a particular cross validation rotation.')
    parser.add_argument('--input-data',
                        dest='input_data',
                        type=str,
                        default='data/raw/TMHMM3.train.3line.latest',
                        help='Path of input data file.')
    parser.add_argument('--pre-trained-model-paths',
                        dest='pre_trained_model_paths',
                        type=str,
                        default=None,
                        help='Paths of pre-trained models.')
    parser.add_argument('--type-predictor-model-path',
                        dest='type_predictor_model_path',
                        type=str,
                        default=None,
                        help='Path of pre-trained type model.')
    parser.add_argument('--profile-path', dest='profile_path',
                        type=str, default="",
                        help='Profiles to use for embedding.')
    parser.add_argument('--embedding', dest='embedding',
                        type=str, default="BLOSUM62",
                        help='Embedding.')
    parser.add_argument('--add-test-to-train', dest='add_test_to_train',
                        action='store_true', default=False,
                        help='Add test data to train.')
    parser.add_argument('--marg', dest='marg',
                        action='store_true', default=False,
                        help='Use marginal probability for decoding.')
    parser.add_argument('--distribute-val', dest='distribute_val',
                        action='store_true', default=False,
                        help='Distribute the validation onto multiple GPUs')
    parser.add_argument('--generate-esm-embeddings', dest='generate_esm_embeddings',
                        action='store_true', default=False)
    parser.add_argument('--esm-embeddings-dir', dest='esm_embeddings_dir',
                        type=str, default='/mnt/embeddings')
    parser.add_argument('--experiment-tag', dest='experiment_tag',
                        type=str, default='')
    args, _unknown = parser.parse_known_args()

    result_matrices = np.zeros((6, 6), dtype=np.int64)

    if args.profile_path != "":
        embedding = "PROFILE"
    else:
        embedding = args.embedding

    if args.embedding == "ESM":
        args.hidden_size = int(1280 / 2)            
        
    use_marg_prob = args.marg
    all_prediction_data = []
    experiment_path = "output/experiments/" + args.experiment_tag
    experiment_file_path = f"{experiment_path}/experiment.json"

    if args.experiment_tag:
        if os.path.exists(experiment_path):
            print("Already ran experiment with this tag! Exiting...")
            sys.exit(1)
        else:
            os.makedirs(experiment_path, exist_ok=True)
        experiment_file_path = f'{experiment_path}/experiment.json'
        open(experiment_file_path, 'w').write(json.dumps(initial_experiment_data_json))
        write_out(f"Running with experiment tag: {args.experiment_tag}")

    for cv_partition in [0, 1, 2, 3, 4]:
        
        # prepare data sets
        train_set, val_set, test_set = load_data_from_disk(filename=args.input_data,
                                                           partition_rotation=cv_partition)

        if args.add_test_to_train:
            train_set = train_set + test_set
            test_set = []

        # topology data set
        train_set_tm = list(filter(lambda x: x[3] == 0, train_set))
        val_set_tm = list(filter(lambda x: x[3] == 0, val_set))
        train_set_sptm = list(filter(lambda x: x[3] == 1, train_set))
        val_set_sptm = list(filter(lambda x: x[3] == 1, val_set))
        train_set_beta = list(filter(lambda x: x[3] == 4, train_set))
        val_set_beta = list(filter(lambda x: x[3] == 4, val_set))

        if not args.silent:
            print("Loaded ",
                  len(train_set), "training,",
                  len(val_set), "validation and",
                  len(test_set), "test samples")

        print("Processing data...")
        pre_processed_path = "data/preprocessed/preprocessed_data_" + str(
            hashlib.sha256(args.input_data.encode()).hexdigest())[:8] + "_cv" \
                             + str(cv_partition) + ".pickle"
        if not os.path.isfile(pre_processed_path):
            input_data_processed = list([TMDataset.from_disk(set, use_gpu) for set in
                                         [train_set, val_set, test_set,
                                          train_set_tm, val_set_tm,
                                          train_set_sptm, val_set_sptm,
                                          train_set_beta, val_set_beta,
                                          ]])
            pickle.dump(input_data_processed, open(pre_processed_path, "wb"))
        input_data_processed = pickle.load(open(pre_processed_path, "rb"))
        train_preprocessed_set = input_data_processed[0]
        validation_preprocessed_set = input_data_processed[1]
        test_preprocessed_set = input_data_processed[2]
        train_preprocessed_set_tm = input_data_processed[3]
        validation_preprocessed_set_tm = input_data_processed[4]
        train_preprocessed_set_sptm = input_data_processed[5]
        validation_preprocessed_set_sptm = input_data_processed[6]
        train_preprocessed_set_beta = input_data_processed[7]
        validation_preprocessed_set_beta = input_data_processed[8]
        
        if args.distribute_val:
            os.makedirs(distributed_data_path, exist_ok=True)
            torch.save(validation_preprocessed_set, open( f"{distributed_data_path}/val_set.pt", "wb")) 
        
        print("Completed preprocessing of data...")

        train_loader = tm_contruct_dataloader_from_disk(train_preprocessed_set,
                                                        args.minibatch_size,
                                                        balance_classes=True)
        validation_loader = tm_contruct_dataloader_from_disk(validation_preprocessed_set,
                                                             args.minibatch_size_validation,
                                                             balance_classes=False)
        test_loader = tm_contruct_dataloader_from_disk(
            test_preprocessed_set if args.evaluate_on_test else validation_preprocessed_set,
            args.minibatch_size_validation)
        
        print(args.embedding, args.generate_esm_embeddings)
        
        if args.embedding == "ESM" and args.generate_esm_embeddings:
            print("Generating Embeddings")
            generate_esm_embeddings(input_data_processed, args.esm_embeddings_dir)
            sys.exit(0)
            

        type_predictor_model_path = args.type_predictor_model_path

        if args.pre_trained_model_paths is None:
            for (experiment_id, train_data, validation_data) in [
                    ("TRAIN_TOPOLOGY_CV" + str(cv_partition)
                     + "-HS" + str(args.hidden_size) + "-F" + str(args.input_data.split(".")[-2])
                     + "-P" + str(args.profile_path.split("_")[-1]) + "-" + embedding, train_loader,
                     validation_loader),
            ]:
                model = TMHMM3(
                    embedding,
                    args.hidden_size,
                    use_gpu,
                    use_marg_prob,
                    None,
                    args.profile_path,
                    args.esm_embeddings_dir,
                )
                    
                model_path = train_model(data_set_identifier=experiment_id,
                                         model=model,
                                         train_loader=train_data,
                                         validation_loader=validation_data,
                                         learning_rate=args.learning_rate,
                                         minibatch_size=args.minibatch_size,
                                         eval_interval=args.eval_interval,
                                         hide_ui=args.hide_ui,
                                         use_gpu=use_gpu,
                                         minimum_updates=args.minimum_updates,
                                         distribute_val=args.distribute_val,
                                         experiment_tag=args.experiment_tag,
                                         cv=cv_partition)

                # let the GC collect the model
                del model

                write_out(model_path)

        else:
            # use the pre-trained model
            model_path = args.pre_trained_model_paths.split(",")[cv_partition]

        # test model
        write_out("Testing model...")
        if not args.evaluate_on_test:
            write_out("Using Validation for test")
            
        if args.experiment_tag:
            prediction_data = test_experiment_and_write_results_to_file(
                cv=str(cv_partition),
                experiment_file_path=experiment_file_path,
                test_loader=test_loader
            )
            all_prediction_data.append(post_process_prediction_data(prediction_data))
            
        else:
            write_out("No experiment tag, so not testing")


    if args.experiment_tag:
        experiment_json = json.loads(open(experiment_file_path, 'r').read())
        write_out(json.dumps(experiment_json, indent=4))
        predictions_file = open(f'{experiment_path}/predictions.txt', 'w')
        predictions_file.write("\n".join(all_prediction_data))
        predictions_file.close()
    else:
        write_out("No experiment tag, so not printing testing results")
    
