import os
import time
import math
import random
from threading import Lock
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score)
from flask import Flask, render_template, request, jsonify, session

PLOT_THEME = {
    'background': '#F5F6FA',
    'paper_bgcolor': '#FFFFFF',
    'plot_bgcolor': '#FFFFFF',
    'font': {'color': '#2D3436', 'family': 'Inter, sans-serif'},
    'gridcolor': '#E2E8F0'
}

app = Flask(__name__, static_folder='static')
app.secret_key = 'ids_dashboard_secret_key'  # For session management

DEMO_EVENT_LOG = []
DEMO_EVENT_LOCK = Lock()
DEMO_EVENT_LIMIT = 100

DEMO_SCENARIOS = {
    'dos': 'DoS',
    'probe': 'Probe / Reconnaissance',
    'r2l': 'Remote to Local (R2L)',
    'u2r': 'User to Root (U2R)',
    'brute_force': 'Brute Force'
}

DEMO_SCENARIO_CLASS_MAP = {
    'nsl_kdd': {
        'dos': 0,
        'probe': 2,
        'r2l': 3,
        'u2r': 4
    },
    'cicids2017': {
        'dos': 1,
        'probe': 3,
        'r2l': 4,
        'brute_force': 0
    }
}

# Configuration for dataset and model paths (adjust as needed)
DATA_PATHS = {
    'cicids2017': 'cicids2017/cic_test.csv',
    'unsw_nb15': 'unsw_nb15/unsw_test.csv',
    'nsl_kdd': 'nsl_kdd/nsl_test.csv'
}

MODEL_PATHS = {
    'cicids2017': 'cicids2017/cic_xgb.pkl',
    'unsw_nb15': 'unsw_nb15/unsw_xgb.pkl',
    'nsl_kdd': 'nsl_kdd/nsl_xgb.pkl'
}

CLASS_MAPPINGS = {
    'nsl_kdd': {
        0: 'Denial of Service (DoS)',
        1: 'Normal',
        2: 'Reconnaissance',
        3: 'Remote to Local (R2L)',
        4: 'User to Root (U2R)'
    },
    'cicids2017': {
        0: 'Brute Force',
        1: 'Denial of Service (DoS/DDoS and Botnet)',
        2: 'Normal',
        3: 'Reconnaissance',
        4: 'Remote to Local (R2L)',
        5: 'Web Attack'
    },
    'unsw_nb15': {
        0: 'Analysis',
        1: 'Backdoor',
        2: 'DoS',
        3: 'Exploits',
        4: 'Fuzzers',
        5: 'Generic',
        6: 'Normal',
        7: 'Reconnaissance',
        8: 'Shellcode',
        9: 'Worms'
    }
}

DATASET_DESCRIPTIONS = {
    'nsl_kdd': "NSL-KDD is used as a benchmark dataset for IDS research with a balanced set of features and attack types.",
    'cicids2017': "CICIDS2017 is a comprehensive dataset with a wide range of attack vectors and a large number of samples for realistic network intrusion detection.",
    'unsw_nb15': "UNSW-NB15 provides diverse network traffic capturing both normal and malicious behaviors with detailed feature sets."
}


def get_target_col(dataset):
    return 'attack_type' if dataset in ['nsl_kdd', 'cicids2017'] else 'attack_cat'


def load_model(dataset):
    model_path = MODEL_PATHS[dataset]
    try:
        return joblib.load(model_path)
    except Exception:
        with open(model_path, 'rb') as file_handle:
            return pickle.load(file_handle)


def get_demo_feature_frame(dataset, target_class=None):
    df = pd.read_csv(DATA_PATHS[dataset])
    target_col = get_target_col(dataset)

    if target_class is not None and target_class in df[target_col].values:
        sample_row = df[df[target_col] == target_class].iloc[0]
    else:
        sample_row = df.iloc[0]

    feature_frame = pd.DataFrame([sample_row.drop(labels=[target_col])])
    return feature_frame, int(sample_row[target_col])


def add_demo_event(event):
    with DEMO_EVENT_LOCK:
        DEMO_EVENT_LOG.append(event)
        if len(DEMO_EVENT_LOG) > DEMO_EVENT_LIMIT:
            del DEMO_EVENT_LOG[:-DEMO_EVENT_LIMIT]


def get_demo_summary():
    with DEMO_EVENT_LOCK:
        recent_events = list(DEMO_EVENT_LOG)

    counts = {}
    for event in recent_events:
        label = event.get('predicted_label', 'Unknown')
        counts[label] = counts.get(label, 0) + 1

    return {
        'events': recent_events[-20:],
        'counts': counts,
        'total_events': len(recent_events)
    }


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/load_dataset', methods=['POST'])
def load_dataset():
    session.clear()
    
    dataset = request.form.get('dataset')
    if not dataset or dataset not in DATA_PATHS:
        return jsonify({'status': 'error', 'message': 'Invalid dataset'})
    try:
        df = pd.read_csv(DATA_PATHS[dataset])
        total_samples = len(df)
        feature_count = df.shape[1] - 1
        
        target_col = get_target_col(dataset)
        
        # Initialize empty line chart data
        line_data = {
            'x': [0],
            'y': [0],
            'type': 'scatter',
            'name': 'Records per Batch'
        }
        
        # Initialize empty pie chart data
        class_mapping = CLASS_MAPPINGS[dataset]
        pie_data = {
            'labels': list(class_mapping.values()),
            'values': [0] * len(class_mapping),
            'type': 'pie'
        }
        
        # Store dataset details in session
        session['dataset'] = dataset
        session['total_samples'] = total_samples
        session['processed_so_far'] = 0
        session['batch_records'] = []
        
        # Calculate batch size and delays
        min_percent = 0.04
        max_percent = 0.16
        batch_percentage = random.uniform(min_percent, max_percent)
        batch_size = math.ceil(total_samples * batch_percentage)
        n_batches = math.ceil(total_samples / batch_size)
        total_delay = min(45, max(10, n_batches * 1.5))
        delay_per_batch = total_delay / n_batches
        
        session['batch_size'] = batch_size
        session['delay_per_batch'] = delay_per_batch
        session['class_distribution'] = {str(k): 0 for k in class_mapping.keys()}

        return jsonify({
            'status': 'success',
            'total_samples': total_samples,
            'feature_count': feature_count,
            'description': DATASET_DESCRIPTIONS.get(dataset, ""),
            'initial_charts': {
                'line_chart': line_data,
                'pie_chart': pie_data
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/process_batch', methods=['POST'])
def process_batch():
    dataset = session.get('dataset')
    if not dataset:
        return jsonify({'status': 'error', 'message': 'No dataset loaded'})
    
    try:
        processed_so_far = int(session.get('processed_so_far', 0))
        total_samples = int(session.get('total_samples'))
        delay_per_batch = float(session.get('delay_per_batch'))
        batch_records = session.get('batch_records', [])

        # Calculate random batch size based on total dataset size
        min_percent, max_percent = 0.04, 0.16
        batch_percentage = random.uniform(min_percent, max_percent)
        batch_size = math.ceil(total_samples * batch_percentage)
        
        # Adjust batch size if near the end of dataset
        remaining = total_samples - processed_so_far
        actual_batch_size = min(batch_size, remaining)
        
        df = pd.read_csv(DATA_PATHS[dataset])
        end_idx = min(processed_so_far + actual_batch_size, total_samples)
        batch_df = df.iloc[processed_so_far:end_idx]
        
        target_col = get_target_col(dataset)
        current_features = batch_df.drop(columns=[target_col]).columns.tolist()[:5]
        
        model = load_model(dataset)
        
        X_batch = batch_df.drop(columns=[target_col])
        y_batch = batch_df[target_col]
        y_pred = model.predict(X_batch)
        
        # Update class distribution
        class_mapping = CLASS_MAPPINGS[dataset]
        current_dist = session.get('class_distribution', {})
        batch_dist = y_batch.value_counts().to_dict()
        for k, v in batch_dist.items():
            current_dist[str(k)] = current_dist.get(str(k), 0) + v
        session['class_distribution'] = current_dist
        
        batch_records.append(len(batch_df))
        session['batch_records'] = batch_records
        
        # Calculate current batch class distribution
        batch_class_dist = pd.Series(y_batch).value_counts().sort_index()
        batch_class_dist.index = [class_mapping[int(i)] for i in batch_class_dist.index]
        
        # Create batch distribution bar chart
        batch_dist_data = {
            'x': batch_class_dist.index.tolist(),
            'y': batch_class_dist.values.tolist(),
            'type': 'bar',
            'name': f'Batch {len(batch_records)}',
            'text': [f'{v:,.0f}' for v in batch_class_dist.values],
            'textposition': 'auto',
            'marker': {
                'color': ['#0984E3', '#00B894', '#6C5CE7', '#FD79A8', '#FDCB6E', '#636E72']
            }
        }

        # Create chart data
        line_data = {
            'x': list(range(1, len(batch_records) + 1)),
            'y': batch_records,
            'type': 'scatter',
            'name': 'Records per Batch'
        }

        pie_data = {
            'labels': [class_mapping[int(k)] for k in sorted(current_dist.keys())],
            'values': [current_dist[k] for k in sorted(current_dist.keys())],
            'type': 'pie'
        }
        
        # Generate classification report
        unique_classes = sorted(set(y_batch.unique()) | set(y_pred))
        class_names = [class_mapping[int(i)] for i in unique_classes]
        report = classification_report(
            y_batch, 
            y_pred, 
            output_dict=True,
            labels=unique_classes,
            target_names=class_names
        )

        processed_so_far = end_idx
        session['processed_so_far'] = processed_so_far
        
        time.sleep(delay_per_batch)
        percentage = round((processed_so_far / total_samples) * 100, 1)
        
        status = 'complete' if processed_so_far >= total_samples else 'in_progress'
        
        # Enhanced batch information
        batch_info = {
            'size': len(batch_df),
            'percentage': round(batch_percentage * 100, 1),
            'total_processed_percentage': round((processed_so_far / total_samples) * 100, 1),
            'remaining': total_samples - end_idx,
            'batch_number': len(batch_records) + 1
        }

        return jsonify({
            'status': status,
            'processed': processed_so_far,
            'total': total_samples,
            'percentage': percentage,
            'current_features': current_features,
            'batch_info': batch_info,
            'line_chart': line_data,
            'pie_chart': pie_data,
            'batch_distribution': batch_dist_data,
            'classification_report': report,
            'batch_number': len(batch_records)
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())  # For server-side debugging
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        })

def create_confusion_matrix_plot(cm, class_names):
    """Create an enhanced confusion matrix plot"""
    fig = go.Figure(data=go.Heatmap(
        z=cm.tolist(),
        x=class_names,
        y=class_names,
        text=cm.tolist(),
        texttemplate="%{text}",
        textfont={"size": 12, "family": "Inter, sans-serif"},
        colorscale=[[0, '#F5F6FA'], [1, '#0984E3']],
        showscale=False,
        hoverongaps=False
    ))
    
    fig.update_layout(
        title={'text': 'Confusion Matrix', 'font': {'size': 20, 'family': 'Inter, sans-serif'}},
        xaxis_title='Predicted Class',
        yaxis_title='Actual Class',
        xaxis={'tickangle': -45},
        width=600,
        height=600,
        margin=dict(t=50, l=100, r=50, b=100),
        paper_bgcolor=PLOT_THEME['paper_bgcolor'],
        plot_bgcolor=PLOT_THEME['plot_bgcolor'],
        font=PLOT_THEME['font']
    )
    return fig

def create_metrics_bar_plot(metrics_dict):
    """Create a bar plot for model metrics"""
    fig = go.Figure(data=[
        go.Bar(
            x=list(metrics_dict.keys()),
            y=list(metrics_dict.values()),
            text=[f'{v:.2%}' for v in metrics_dict.values()],
            textposition='auto',
            marker_color='#0984E3',
            hoverinfo='y'
        )
    ])
    fig.update_layout(
        title={'text': 'Model Performance Metrics', 'font': {'size': 20}},
        yaxis_title='Score',
        xaxis_title='Metric',
        xaxis={'tickangle': -45},
        yaxis={'range': [0, 1], 'gridcolor': PLOT_THEME['gridcolor']},
        width=700,
        height=500,
        margin=dict(t=50, l=50, r=50, b=100),
        paper_bgcolor=PLOT_THEME['paper_bgcolor'],
        plot_bgcolor=PLOT_THEME['plot_bgcolor'],
        font=PLOT_THEME['font']
    )
    return fig

def create_class_distribution_bar(class_dist):
    """Create a bar plot for class distribution"""
    fig = go.Figure(data=[
        go.Bar(
            x=class_dist.index,
            y=class_dist.values,
            text=[f'{v:,.0f}' for v in class_dist.values],
            textposition='auto',
            marker_color=['#0984E3', '#00B894', '#6C5CE7', '#FD79A8', '#FDCB6E', '#636E72'],
            hoverinfo='x+y'
        )
    ])
    
    fig.update_layout(
        title={'text': 'Class Distribution', 'font': {'size': 20}},
        yaxis_title='Number of Samples',
        xaxis_title='Class',
        xaxis={'tickangle': -45},
        height=500,
        margin=dict(t=50, l=50, r=50, b=100),
        paper_bgcolor=PLOT_THEME['paper_bgcolor'],
        plot_bgcolor=PLOT_THEME['plot_bgcolor'],
        font=PLOT_THEME['font']
    )
    return fig

@app.route('/final_evaluation', methods=['POST'])
def final_evaluation():
    # Run a comprehensive evaluation on the full dataset once all batches are processed
    dataset = session.get('dataset')
    if not dataset:
        return jsonify({'status': 'error', 'message': 'No dataset loaded'})
    try:
        df = pd.read_csv(DATA_PATHS[dataset])
        target_col = get_target_col(dataset)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        model = load_model(dataset)
            
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None

        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y, y_pred)
        if y_pred_proba is not None:
            if len(np.unique(y)) == 2:
                roc_auc = roc_auc_score(y, y_pred_proba[:, 1])
            else:
                roc_auc = roc_auc_score(pd.get_dummies(y), y_pred_proba, multi_class='ovr', average='macro')
        else:
            roc_auc = None

        # Create a confusion matrix plot
        class_mapping = CLASS_MAPPINGS[dataset]
        sorted_keys = sorted(class_mapping.keys())
        class_names = [class_mapping[i] for i in sorted_keys]
        cm = confusion_matrix(y, y_pred)
        cm_plot = create_confusion_matrix_plot(cm, class_names)

        # Create overall metrics bar plot
        metrics_dict = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'MCC': mcc,
            'ROC AUC': roc_auc if roc_auc is not None else 0
        }
        metrics_plot = create_metrics_bar_plot(metrics_dict)

        # Create class distribution plots
        class_dist = pd.Series(y).value_counts().sort_index()
        class_dist.index = [class_mapping.get(i, str(i)) for i in class_dist.index]
        
        # Bar chart for class distribution
        dist_bar = create_class_distribution_bar(class_dist)
        
        # Existing pie chart
        pie_fig = go.Figure(data=[
            go.Pie(
                labels=class_dist.index,
                values=class_dist.values,
                textinfo='label+percent',
                marker=dict(colors=['#0984E3', '#00B894', '#6C5CE7', '#FD79A8', '#FDCB6E', '#636E72'])
            )
        ])
        pie_fig.update_layout(
            title={'text': 'Class Distribution (Pie)', 'font': {'size': 20}},
            paper_bgcolor=PLOT_THEME['paper_bgcolor'],
            plot_bgcolor=PLOT_THEME['plot_bgcolor'],
            font=PLOT_THEME['font']
        )

        # Generate ROC curve or Precision-Recall curve if possible (using Plotly Express)
        # For brevity, we produce a ROC curve for binary classification; extend as needed.
        roc_fig = None
        if y_pred_proba is not None and len(np.unique(y)) == 2:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y, y_pred_proba[:, 1])
            roc_fig = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines'))
            roc_fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')

        # Generate a comprehensive classification report
        unique_classes = sorted(set(y.unique()) | set(y_pred))
        class_names = [class_mapping[int(i)] for i in unique_classes]
        report = classification_report(
            y, y_pred,
            output_dict=True,
            labels=unique_classes,
            target_names=class_names
        )

        return jsonify({
            'status': 'success',
            'metrics': metrics_dict,
            'confusion_matrix_plot': cm_plot.to_json(),
            'metrics_plot': metrics_plot.to_json(),
            'pie_chart': pie_fig.to_json(),
            'dist_bar_chart': dist_bar.to_json(),
            'roc_curve': roc_fig.to_json() if roc_fig is not None else None,
            'classification_report': report,
            'final_values': {
                'Accuracy': round(accuracy, 3),
                'ROC AUC': round(roc_auc, 3) if roc_auc is not None else None,
                'F1 Score': round(f1, 3),
                'MCC': round(mcc, 3)
            }
        })
    except Exception as e:
        import traceback
        return jsonify({'status': 'error', 'message': str(e), 'traceback': traceback.format_exc()})
@app.route('/demo_emit', methods=['POST'])
def demo_emit():
    payload = request.get_json(silent=True) or request.form.to_dict()
    dataset = payload.get('dataset') or session.get('dataset') or 'cicids2017'
    scenario = payload.get('scenario', '').strip().lower()
    source_ip = payload.get('source_ip', 'attacker')
    destination_ip = payload.get('destination_ip', 'defender')

    if dataset not in MODEL_PATHS or dataset not in DEMO_SCENARIO_CLASS_MAP:
        return jsonify({'status': 'error', 'message': 'Unsupported dataset for demo mode'})

    scenario_map = DEMO_SCENARIO_CLASS_MAP[dataset]
    if scenario not in scenario_map:
        return jsonify({'status': 'error', 'message': 'Unsupported demo scenario for selected dataset'})

    target_class = scenario_map[scenario]
    model = load_model(dataset)
    feature_frame, source_class = get_demo_feature_frame(dataset, target_class)
    prediction = model.predict(feature_frame)[0]
    predicted_class = int(prediction)
    class_mapping = CLASS_MAPPINGS[dataset]

    predicted_label = class_mapping.get(predicted_class, str(predicted_class))
    expected_label = class_mapping.get(target_class, str(target_class))

    confidence = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(feature_frame)[0]
        confidence = float(np.max(probabilities))

    event = {
        'timestamp': time.time(),
        'dataset': dataset,
        'scenario': scenario,
        'expected_label': expected_label,
        'source_label': class_mapping.get(source_class, str(source_class)),
        'predicted_label': predicted_label,
        'predicted_class': predicted_class,
        'confidence': confidence,
        'source_ip': source_ip,
        'destination_ip': destination_ip
    }

    add_demo_event(event)

    # Return minimal info to attacker - no detection feedback
    return jsonify({
        'status': 'success',
        'message': 'Attack pattern received',
        'timestamp': event['timestamp']
    })


@app.route('/demo_events', methods=['GET'])
def demo_events():
    return jsonify({'status': 'success', 'summary': get_demo_summary()})


@app.route('/demo_clear', methods=['POST'])
def demo_clear():
    with DEMO_EVENT_LOCK:
        DEMO_EVENT_LOG.clear()
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    # Ensure the required directories exist (if not already present)
    for dir_name in ['cicids2017', 'unsw_nb15', 'nsl_kdd']:
        os.makedirs(dir_name, exist_ok=True)
    app.run(debug=True)
