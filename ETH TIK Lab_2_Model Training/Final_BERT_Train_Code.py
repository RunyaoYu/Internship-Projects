"""
Author: Runyao Yu
runyao.yu@tum.de
Research Internship in ETH Zurich
For Academic Use Purpose only
"""

# Input data files are available in the "../data/" directory.
import pickle
import os

# Basics + Viz
import time
import datetime
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns

# Pre-processing
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences

# Models
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup

# Metrics
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# Suppress warnings
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


if __name__ == '__main__':

    batch_size = 1
    epochs = 5
    # topics = ["和博主互动", "想要认识对方", "想要获得更多信息", "表达情感见解看法，分享过往经历"]
    topics = ["表达情感见解看法，分享过往经历"]

    df = pd.read_excel('Expressing emotional opinions and sharing experiences.xlsx')
    # df = df_origin[:50]
    text = df['content'].tolist()
    df = df.drop(['content'], axis=1)
    df = df.fillna(0)
    # df = df.apply(lambda x: pd.factorize(x)[0])
    df['text'] = text

    print('Before processing:')
    # print("在 label 列中总共有 %d 个空值." % df['label'].isnull().sum())
    print("在 answer 列中总共有 %d 个空值." % df['text'].isnull().sum())
    print('\n')
    df[df.isnull().values == True]
    # df = df[pd.notnull(df['label'])]
    df = df[pd.notnull(df['text'])]
    print('After processing:')
    # print("在 label  列中总共有 %d 个空值." % df['label'].isnull().sum())
    print("在 answer 列中总共有 %d 个空值." % df['text'].isnull().sum())
    print('\n')
    # df.info()
    print('\n')

    temp = df.drop(['text'], axis=1)
    for (columnName, columnData) in temp.iteritems():
        d = {'{}'.format(columnName): df['{}'.format(columnName)].value_counts().index,
             'count': df['{}'.format(columnName)].value_counts()}
        df_label = pd.DataFrame(data=d).reset_index(drop=True)

    # prepare GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda:0")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)  # 512max

    MAX_LEN = 79

    sentence_lengths = []


    def tokenize_and_count(s, lst, max_len):
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        answer = tokenizer.encode(s, add_special_tokens=True)
        lst.append(len(answer))

        return answer


    df['bert'] = df.text.apply(lambda s: tokenize_and_count(s, sentence_lengths, MAX_LEN))
    df['bert_aug'] = df.text.apply(lambda s: tokenize_and_count(s, sentence_lengths, MAX_LEN))

    # Pad our input tokens with value 0.
    # "post" indicates that we want to pad and truncate at the end of the sequence,
    # as opposed to the beginning.

    df['bert'] = pad_sequences(df['bert'].values, maxlen=MAX_LEN, dtype="long",
                               value=0, truncating="post", padding="post").tolist()
    df['bert_aug'] = pad_sequences(df['bert_aug'].values, maxlen=MAX_LEN, dtype="long",
                                   value=0, truncating="post", padding="post").tolist()

    # Attention Masks
    # Create attention masks
    df['attention'] = df['bert'].apply(lambda arr: [int(token_id > 0) for token_id in arr])
    df['attention_aug'] = df['bert_aug'].apply(lambda arr: [int(token_id > 0) for token_id in arr])

    test_size = 0.2
    train_df, test_df = train_test_split(df, random_state=42, test_size=test_size)
    print(f"{test_size} split\n{train_df.shape[0]} lines of training data,\n{test_df.shape[0]} lines of test data")


    # Define some util functions

    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        try:
            roc = roc_auc_score(pred_flat, labels_flat)
        except ValueError:
            roc = 0
        return f1_score(pred_flat, labels_flat), roc, np.sum(pred_flat == labels_flat) / len(labels_flat)


    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))


    def single_topic_train(model, epochs, train_dataloader, test_dataloader, seed_val=42, verbose=False):
        # This training code is based on the `run_glue.py` script here:

        # Set the seed value all over the place to make this reproducible.
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        # torch.cuda.manual_seed_all(seed_val)

        # Store the average loss after each epoch so we can plot them.
        training_losses = []
        testing_losses = []

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)  # device_ids=[0,1]

        model.to(device)

        for epoch_i in range(0, epochs):
            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.
            if verbose:
                print("")
                print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
                print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_train_loss = 0

            # Put the model into training mode.
            model.train()

            # For each batch of training data...
            for step, batch in enumerate(train_dataloader):

                # b_input_ids = batch[0].to(f'cuda:{model.device_ids[0]}')
                # b_input_mask = batch[1].to(f'cuda:{model.device_ids[0]}')
                # b_labels = batch[2].to(f'cuda:{model.device_ids[0]}')
                b_input_ids = batch[0].to(device)  # to(device)
                b_input_mask = batch[1].to(device)  # to(device)
                b_labels = batch[2].to(device)  # to(device)

                if step % 40 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)

                    # Report progress.
                    if verbose:
                        print(f'  Batch {step:>5,}  of  {len(train_dataloader):>5,}.    Elapsed: {elapsed:}.')

                model.zero_grad()

                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask,
                                labels=b_labels)
                # print("Output长什么样:\n")
                # print(outputs)
                # print("-" * 10)
                loss = outputs[0]
                # print(loss.mean())
                total_train_loss += float(loss.mean())
                # total_train_loss += outputs[0].item()

                loss.mean().backward()
                # loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over the training data.
            avg_train_loss = total_train_loss / len(train_dataloader)
            training_losses.append(avg_train_loss)

            if verbose:
                print("")
                print("  Average training loss: {0:.2f}".format(avg_train_loss))
                print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            if verbose:
                print("")
                print("Running Validation...")

            # Measure how long the testing takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_test_loss = 0

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            # torch.cuda.empty_cache()
            model.eval()

            # Tracking variables
            eval_loss, eval_accuracy, eval_f1, eval_auc = 0, 0, 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # Evaluate data for one epoch
            for batch in test_dataloader:
                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)

                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch

                # Telling the model not to compute or store gradients, saving memory and
                # speeding up validation
                with torch.no_grad():
                    outputs = model(b_input_ids,
                                    token_type_ids=None,
                                    attention_mask=b_input_mask,
                                    labels=b_labels)

                total_test_loss += outputs[0].mean()

                logits = outputs[1]
                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences.
                tmp_eval_f1, tmp_eval_auc, tmp_eval_accuracy = flat_accuracy(logits, label_ids)

                # Accumulate the total accuracy.
                eval_accuracy += tmp_eval_accuracy
                eval_f1 += tmp_eval_f1
                eval_auc += tmp_eval_auc
                # Track the number of batches
                nb_eval_steps += 1

            avg_test_loss = total_test_loss / len(test_dataloader)
            testing_losses.append(avg_test_loss)

            # Report the final accuracy for this validation run.
            if verbose:
                print("  F1 Score: {0:.2f}".format(eval_f1 / nb_eval_steps))
                print("  ROC_AUC: {0:.2f}".format(eval_auc / nb_eval_steps))
                print("  Accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
                print("  Validation took: {:}".format(format_time(time.time() - t0)))
                print("  Average validation loss: {0:.2f}".format(avg_test_loss))

        return model, training_losses, testing_losses


    # Evaluation
    def run_evaluation(model, test_dataloader, verbose=False):
        # Put model in evaluation mode
        # torch.cuda.empty_cache()
        model.eval()

        # Tracking variables
        predictions, true_labels = [], []

        # Predict
        for batch in test_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask)

            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)

        # Create results
        matthews_set = []

        # Evaluate each test batch using Matthew's correlation coefficient
        if verbose:
            print('Calculating Matthews Corr. Coef. for each batch...')

        # For each input batch...
        for i in range(len(true_labels)):
            # The predictions for this batch are a 2-column ndarray (one column for "0"
            # and one column for "1"). Pick the label with the highest value and turn this
            # in to a list of 0s and 1s.
            pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
            # Calculate and store the coef for this batch.
            matthews = matthews_corrcoef(true_labels[i], pred_labels_i)

            if verbose:
                print("Predicted Label for Batch " + str(i) + " is " + str(pred_labels_i))
                print("True Label for Batch " + str(i) + " is " + str(true_labels[i]))
                print("Matthew's correlation coefficient for Batch " + str(i) + " is " + str(matthews))
            matthews_set.append(matthews)

        # Combine the predictions for each batch into a single list of 0s and 1s.
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

        # Combine the correct labels for each batch into a single list.
        flat_true_labels = [item for sublist in true_labels for item in sublist]

        # Calculate the MCC
        mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
        f1 = f1_score(flat_true_labels, flat_predictions)
        ra = roc_auc_score(flat_true_labels, flat_predictions)

        cm = confusion_matrix(flat_true_labels, flat_predictions)
        sns.heatmap(cm, annot=True)
        plt.show()

        print('MCC: %.3f' % mcc)
        print('ROC_AUC: %.3f' % ra)
        print('F1: %.3f' % f1)
        print(classification_report(flat_true_labels, flat_predictions))


    # Run Model
    def augmented_dataloader(train_df, test_df, topic, batch_size):
        # Test data should NOT have any augmented data in it.
        # Therefore the process is the same as the past.
        test_x = torch.tensor(test_df.bert.values.tolist())
        test_y = torch.tensor(test_df[topic].values.astype(int))
        test_masks = torch.tensor(test_df.attention.values.tolist())

        test_data = TensorDataset(test_x, test_masks, test_y)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size, pin_memory=True)

        # Train data should have the augmented data added.
        # BUT we only want the positive cased augmented. No need in throwing in more negative cases.
        aug_index = train_df[topic] == 1

        train_x = train_df.bert.values.tolist() + train_df.bert_aug[aug_index].values.tolist()
        train_y = train_df[topic].values.astype(int).tolist() + train_df[topic][aug_index].values.astype(int).tolist()
        train_masks = train_df.attention.values.tolist() + train_df.attention[aug_index].values.tolist()

        train_x = torch.tensor(train_x)
        train_y = torch.tensor(train_y)
        train_masks = torch.tensor(train_masks)

        train_data = TensorDataset(train_x, train_masks, train_y)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, pin_memory=True)

        return train_dataloader, test_dataloader


    # Create x, y for each
    for topic in topics:
        train_dataloader, test_dataloader = augmented_dataloader(train_df, test_df, topic, batch_size)

        # Then load the pretrained BERT model (has linear classification layer on top)
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=2,  # The number of output labels--2 for binary classification.
            # You can increase this for multi-class tasks.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )

        # model = LongformerForSequenceClassification.from_pretrained(
        #     'allenai/longformer-base-4096')  # longformer 替换上面的bert

        # model.cpu()
        model.cuda(device = device)  # for gpu device=device

        # load optimizer
        optimizer = AdamW(model.parameters(),
                          lr=2e-5,  # args.learning_rate - default is 5e-5 2
                          eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                          )

        # Total number of training steps is [number of batches] x [number of epochs].
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        model, train_losses, test_losses = single_topic_train(model, epochs,
                                                              train_dataloader,
                                                              test_dataloader,
                                                              seed_val=42)

        # Visualize test and train curve
        # draw_test_train_curve(test_losses, train_losses, topic)
        print("====================")
        print('train losses:', train_losses)
        print('test losses:', test_losses)

        # Evaluation results
        print("====================")
        print("====================")
        print("EVALUATION")
        print(f"TOPIC: {topic}")
        run_evaluation(model, test_dataloader, verbose=False)

    # Save Trained Models

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

    output_dir = './Expressing_model_save/'
    print(output_dir)
    # # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epochs}

    # torch.save(state, os.path.join(output_dir, 'training_args.bin'))
    # torch.save(model.state_dict(), output_dir)





