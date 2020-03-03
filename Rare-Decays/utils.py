from sklearn.utils.validation import column_or_1d
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import pandas
from hep_ml import metrics

def check_correlation(probabilities, mass):
    probabilities, mass = map(column_or_1d, [probabilities, mass])

    y_pred = np.zeros(shape=(len(probabilities), 2))
    y_pred[:, 1] = probabilities
    y_pred[:, 0] = 1 - probabilities
    y_true = [0] * len(probabilities)
    df_mass = pandas.DataFrame({'mass': mass})
    cvm = metrics.BinBasedCvM(uniform_features=['mass'], uniform_label=0)
    cvm.fit(df_mass, y_true)
    return cvm(y_true, y_pred, sample_weight=None)

def roc_curve_splitted(data1, data2, sample_weights1, sample_weights2):
    labels = [0] * len(data1) + [1] * len(data2)
    weights = np.concatenate([sample_weights1, sample_weights2])
    data = np.concatenate([data1, data2])
    return roc_curve(labels, data, sample_weight=weights)

def check_agreement_ks_sample_weighted (data_prediction, mc_prediction, weights_data, weights_mc):
    data_prediction, weights_data = map(column_or_1d, [data_prediction, weights_data])
    mc_prediction, weights_mc = map(column_or_1d, [mc_prediction, weights_mc])

    assert np.all(data_prediction >= 0.) and np.all(data_prediction <= 1.), 'error in prediction'
    assert np.all(mc_prediction >= 0.) and np.all(mc_prediction <= 1.), 'error in prediction'

    weights_data = weights_data / np.sum(weights_data)
    weights_mc = weights_mc / np.sum(weights_mc)

    data_neg = data_prediction[weights_data < 0]
    weights_neg = -weights_data[weights_data < 0]
    mc_prediction = np.concatenate((mc_prediction, data_neg))
    weights_mc = np.concatenate((weights_mc, weights_neg))
    data_prediction = data_prediction[weights_data >= 0]
    weights_data = weights_data[weights_data >= 0]

    assert np.all(weights_data >= 0) and np.all(weights_mc >= 0)
    assert np.allclose(weights_data.sum(), weights_mc.sum())

    weights_data /= np.sum(weights_data)
    weights_mc /= np.sum(weights_mc)

    fpr, tpr, _ = roc_curve_splitted(data_prediction, mc_prediction, weights_data, weights_mc)

    Dnm = np.max(np.abs(fpr - tpr))
    Dnm_part = np.max(np.abs(fpr - tpr)[fpr + tpr < 1])

    result = {'ks': Dnm, 'ks_part': Dnm_part}
    return Dnm_part < 0.03, result

def get_ks_metric(df_agree, df_test):
    sig_ind = df_agree[df_agree['signal'] == 1].index
    bck_ind = df_agree[df_agree['signal'] == 0].index

    mc_prob = np.array(df_test.loc[sig_ind]['prediction'])
    mc_weight = np.array(df_agree.loc[sig_ind]['weight'])
    data_prob = np.array(df_test.loc[bck_ind]['prediction'])
    data_weight = np.array(df_agree.loc[bck_ind]['weight'])
    val, agreement_metric = check_agreement_ks_sample_weighted(data_prob, mc_prob, data_weight, mc_weight)
    return agreement_metric['ks']

