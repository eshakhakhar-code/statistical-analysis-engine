from flask import Flask, request, jsonify, render_template_string
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

# --- FRONTEND HTML CODE ---
FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hypothesis Tester</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; max-width: 600px; margin: 40px auto; padding: 20px; background-color: #f9f9fa; color: #333; }
        .card { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h2 { margin-top: 0; color: #2c3e50; }
        label { font-weight: bold; display: block; margin-top: 15px; margin-bottom: 5px; }
        input, select, textarea { width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 5px; box-sizing: border-box; font-size: 16px; }
        button { margin-top: 20px; width: 100%; padding: 12px; background-color: #007aff; color: white; border: none; border-radius: 5px; font-size: 16px; font-weight: bold; cursor: pointer; }
        button:hover { background-color: #005bb5; }
        #result-box { margin-top: 25px; padding: 20px; border-radius: 5px; display: none; background-color: #e8f4fd; border-left: 5px solid #007aff; }
        .error { background-color: #fde8e8; border-left-color: #ff3b30; }
    </style>
</head>
<body>
    <div class="card">
        <h2>📊 Statistical Analysis Engine</h2>
        <p>Enter your dataset below to run a One-Sample T-Test.</p>
        
        <form id="statForm">
            <label>Data Points (comma or space separated):</label>
            <textarea name="data" rows="3" placeholder="e.g. 12.5, 14.1, 11.8, 13.2" required></textarea>

            <label>Population Mean (mu):</label>
            <input type="number" step="any" name="mu" value="0" required>

            <label>Significance Level (Alpha):</label>
            <input type="number" step="any" name="alpha" value="0.05" required>

            <label>Alternative Hypothesis:</label>
            <select name="alternative">
                <option value="two-sided">Two-sided</option>
                <option value="greater">Greater</option>
                <option value="less">Less</option>
            </select>

            <button type="submit">Calculate T-Test</button>
        </form>

        <div id="result-box"></div>
    </div>

    <script>
        document.getElementById('statForm').addEventListener('submit', async function(e) {
            e.preventDefault(); // Prevent page reload
            
            const resultBox = document.getElementById('result-box');
            resultBox.style.display = 'block';
            resultBox.className = ''; 
            resultBox.innerHTML = 'Calculating...';

            const formData = new FormData(this);

            try {
                // Send data to the Python backend
                const response = await fetch('/test', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();

                if (data.error) {
                    resultBox.className = 'error';
                    resultBox.innerHTML = `<strong>Error:</strong> ${data.error}`;
                } else {
                    // Display success results
                    resultBox.innerHTML = `
                        <h3 style="margin-top:0;">Result: ${data.conclusion}</h3>
                        <strong>P-Value:</strong> ${data.p_value} <br>
                        <strong>T-Calculated:</strong> ${data.t_calculated} <br>
                        <strong>Sample Mean:</strong> ${data.mean} <br>
                        <strong>Degrees of Freedom:</strong> ${data.df}
                    `;
                }
            } catch (err) {
                resultBox.className = 'error';
                resultBox.innerHTML = `<strong>Server Error:</strong> Could not connect to the API.`;
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    # This serves the HTML block above directly to the browser
    return render_template_string(FRONTEND_HTML)

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
