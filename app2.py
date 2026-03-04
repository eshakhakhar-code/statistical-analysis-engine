from flask import Flask, request, render_template, jsonify
from scipy.stats import t
import numpy as np
from statistics import stdev
import json
 
app = Flask(__name__)
 
def hypothesis_test_calculation(X, alpha, mu, alternative):
    """
    Perform hypothesis test and return results as dictionary
    """
    x_bar = np.mean(X)
    sd = stdev(X)
    n = len(X)
    df = n - 1
    
    stderror = sd / (n ** 0.5)
    t_cal = (x_bar - mu) / stderror
    
    result = {
        'mean': round(x_bar, 4),
        'stdev': round(sd, 4),
        'n': n,
        'df': df,
        't_calculated': round(t_cal, 4),
        'alpha': alpha,
        'mu': mu,
        'alternative': alternative
    }
    
    if alternative == 'less':
        t_table_neg = t.ppf(alpha, df)
        p_value = t.cdf(t_cal, df)
        
        result['t_table_neg'] = round(t_table_neg, 4)
        result['p_value'] = round(p_value, 4)
        result['conclusion'] = 'Reject H0' if t_cal < t_table_neg else 'Fail to Reject H0'
        result['rejection_region'] = f't < {round(t_table_neg, 4)}'
        
    elif alternative == 'greater':
        t_table_pos = t.ppf(1 - alpha, df)
        p_value = 1 - t.cdf(t_cal, df)
        
        result['t_table_pos'] = round(t_table_pos, 4)
        result['p_value'] = round(p_value, 4)
        result['conclusion'] = 'Reject H0' if t_cal > t_table_pos else 'Fail to Reject H0'
        result['rejection_region'] = f't > {round(t_table_pos, 4)}'
        
    else:  # two-sided
        alpha1 = alpha / 2
        t_table_pos = t.ppf(1 - alpha1, df)
        t_table_neg = t.ppf(alpha1, df)
        p_value = 2 * (1 - t.cdf(abs(t_cal), df))
        
        result['t_table_neg'] = round(t_table_neg, 4)
        result['t_table_pos'] = round(t_table_pos, 4)
        result['p_value'] = round(p_value, 4)
        result['conclusion'] = 'Reject H0' if (t_cal < t_table_neg or t_cal > t_table_pos) else 'Fail to Reject H0'
        result['rejection_region'] = f't < {round(t_table_neg, 4)} or t > {round(t_table_pos, 4)}'
    
    return result
 
@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/test', methods=['POST'])
def test():
    try:
        # Get data from form
        data = request.form.get('data', '')
        alpha = float(request.form.get('alpha', 0.05))
        mu = float(request.form.get('mu', 0))
        alternative = request.form.get('alternative', 'two-sided')
        
        # Parse the data
        if not data:
            return jsonify({'error': 'Please enter data'}), 400
        
        # Handle different input formats
        try:
            # Try parsing as JSON array first
            values = json.loads(f'[{data}]')
        except:
            # If that fails, split by commas or spaces
            if ',' in data:
                values = [float(x.strip()) for x in data.split(',') if x.strip()]
            else:
                values = [float(x) for x in data.split() if x.strip()]
        
        if len(values) < 2:
            return jsonify({'error': 'Please enter at least 2 numbers'}), 400
        
        # Perform hypothesis test
        result = hypothesis_test_calculation(values, alpha, mu, alternative)
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({'error': f'Invalid number format: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
if __name__ == '__main__':
    app.run(debug=True)
 