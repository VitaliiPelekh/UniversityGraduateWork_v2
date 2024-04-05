from flask import Flask, request, render_template, jsonify, redirect, url_for, session, send_file, make_response, current_app
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from joblib import load
from io import BytesIO
import base64
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib.ticker as mticker
import mplfinance as mpf
import seaborn as sns
import json
import os
import tempfile
import uuid
import xlsxwriter
import time
from flask_babel import Babel, gettext as _, lazy_gettext as _l

app = Flask(__name__)
app.secret_key = 'supersecretkey123'


def get_locale():
    return session.get('language', request.accept_languages.best_match(app.config['LANGUAGES'].keys()))


@app.context_processor
def context_processor():
    return dict(get_locale=get_locale)


babel = Babel(app)
babel.init_app(app, locale_selector=get_locale)

# Configure available languages
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_DEFAULT_TIMEZONE'] = 'UTC'
app.config['LANGUAGES'] = {
    'en': 'English',
    'uk': 'Ukrainian'
}

NUM_ASSETS = 10  # The total number of assets expected by the neural network
FEATURES_PER_ASSET = 2  # The number of features per asset (e.g., weight and risk level)
MIN_LENGTH = 803  # The minimum size of close values lists of the assets

font_path = os.path.join(app.root_path, 'static', 'Fonts', 'DejaVuSans.ttf')
font_path_bold = os.path.join(app.root_path, 'static', 'Fonts', 'DejaVuSans-Bold.ttf')
pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))
pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', font_path_bold))

temp_dir = 'temp_data'
lifetime = 3600  # File lifetime in seconds (eg 1 hour)
current_time = time.time()

for filename in os.listdir(temp_dir):
    file_path = os.path.join(temp_dir, filename)
    if os.stat(file_path).st_mtime < current_time - lifetime:
        os.remove(file_path)
        print(f"Deleted file: {file_path}")


@app.route('/set_language/<language>')
def set_language(language):
    if language in app.config['LANGUAGES'].keys():
        session['language'] = language
    return redirect(request.referrer or url_for('index'))


# Initialization of model and scaler
scaler = load('NN models/scaler_best.joblib')
model_FNN = load_model("NN models/ForwardNN_model_best.h5")
prediction_model_SNN = load_model('NN models/SNN_CNN_Pearson_model_real_normalized_best.h5')

asset_risk_classification = {
    'low_risk': {
        'defensive': {
            'bonds': ["TLT", "IEF", "SHY", "GOVT", "IEI", "TLH", "EDV", "BND", "AGG", "LQD"],
            'gold_and_precious_metals': ["GLD", "SLV", "IAU", "SGOL", "SIVR", "PALL", "PLTM", "GDX", "GDXJ", "RING"],
            'stable_sector_etfs': ["VHT", "XLU", "XLV", "VPU", "IYH", "FXG", "FSTA", "JXI", "IDU", "IYZ"]
        },
        'liquid': {
            'money_market_etfs': ["SHV", "BIL", "SPTS", "VGSH", "SCHO", "GBIL", "NEAR", "SHM", "MINT", "FLRN"]
        }
    },
    'medium_risk': {
        'defensive': {
            'bank_sector': ["JPM", "BAC", "WFC", "C", "GS", "MS", "HSBC", "BCS", "CS", "DB"],
            'defense_stocks': ["LMT", "RTX", "NOC", "GD", "LHX", "TXT", "HII", "KTOS", "AJRD"],
            'food_sector_stocks': ["NSRGY", "UL", "K", "GIS", "CPB", "HRL", "TSN", "ADM", "KHC", "MDLZ"],
            'healthcare_stocks': ["JNJ", "UNH", "PFE", "MRK", "ABBV", "AMGN", "GILD", "BMY", "MDT", "SYK", "MRNA", "AZN"],
            'oil_and_gas_stocks': ["XOM", "CVX", "BP", "COP", "EOG", "PXD", "VLO", "MPC"],
            'mining_stocks': ["BHP", "RIO", "FCX", "SCCO", "TECK", "AA", "CLF", "BTU", "ARCH", "HCC"],
        },
        'liquid': {
            'large_stable_stocks': ["PEP", "PG", "KO", "WMT", "MCD", "T", "VZ", "DUK", "SO", "HD"]
        },
        'venture': {
            'innovation_focused_etfs': ["ARKK", "ARKG", "ARKW", "ARKF", "ARKQ", "XITK", "PSI", "BOTZ", "FINX", "CLOU"]
        }
    },
    'high_risk': {
        'speculative': {
            'stocks': ["AAPL", "TSLA", "AMZN", "META", "GOOGL", "NFLX", "NVDA", "AMD", "BA", "BABA", "INTC", "MSFT", "ORCL", "DIS", "IBM", "CSCO", "QCOM", "SHOP", "TWTR", "SNAP", "PLTR", "BYND", "NIO", "CRM", "V"],
            'cryptocurrencies': ["BTCUSD", "ETHUSD", "LTCUSD", "XRPUSD", "LINKUSD", "SOLUSD", "DOTUSD", "ADAUSD", "BCHUSD", "EOSUSD", "XMRUSD", "DASHUSD", "ZECUSD", "BNBUSD"],
            'real_estate_etfs': ["VNQ", "IYR", "SCHH", "RWR", "REM", "MORT", "VNQI", "REZ", "ICF", "XLRE"],
            'developing_markets': ["EEM", "VWO", "IEMG", "BKF", "SCHE", "PIE", "QEMM", "EMXC", "EMIF"],
            'sport_wear': ["NKE", "UAA", "LULU", "VFC", "DECK", "COLM", "GRMN"]
        },
        'venture': {
            'tech_stocks': ["SPCE", "UBER", "LYFT", "ZM", "PTON", "SQ", "ROKU", "SNOW", "DOCU"],
            'biotechnologies': ["REGN", "VRTX", "ALNY", "BLUE", "SGEN", "NTLA", "CRSP", "EDIT", "BMRN", "SRPT"],
            'green_energy': ["RUN", "SEDG", "FSLR", "SPWR", "NEE", "BEP", "CWEN"]
        }
    }
}


def get_all_tickers(asset_risk_classification):
    tickers = []
    for risk_level, asset_classes in asset_risk_classification.items():
        for asset_class, assets in asset_classes.items():
            for asset_type, asset_tickers in assets.items():
                tickers.extend(asset_tickers)
    return tickers


all_tickers = get_all_tickers(asset_risk_classification)
loaded_data = {}
historical_data = {}
historical_data_OHLC = {}

for ticker in all_tickers:
    df = pd.read_csv(f'Asset OHLC data/{ticker}_historical_OHLC_data.csv', skipfooter=1, engine='python')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    loaded_data[ticker] = df

for ticker, df in loaded_data.items():
    df = df.sort_index(ascending=True)
    historical_data[ticker] = df['close'].tolist()
    historical_data_OHLC[ticker] = df[['open', 'high', 'low', 'close', 'volume']]


@app.route('/', methods=['GET'])
def index():
    # Serialize data to JSON strings
    asset_risk_classification_json = json.dumps(asset_risk_classification)
    historical_data_json = json.dumps(historical_data)

    return render_template('create_portfolio.html',
                           asset_risk_classification_json=asset_risk_classification_json,
                           historical_data_json=historical_data_json)


@app.route('/get_asset_data/<ticker>')
def get_asset_data(ticker):
    try:
        current_lang = get_locale()  # 'en' or 'uk'

        df = pd.read_csv(f'Asset Description Translated/{ticker}.csv')
        data = df.to_dict(orient='records')[0]

        # Localization of description
        localized_description_key = f"description_{current_lang}"
        if localized_description_key in data:
            data['localized_description'] = data[localized_description_key]
        else:
            data['localized_description'] = data.get('description', 'No description available')

        for key, value in data.items():
            if pd.isna(value):
                data[key] = None

        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/create_portfolio', methods=['POST'])
def create_portfolio():
    session['initial_investment'] = float(request.form['initial_investment'])
    session['financial_cushion_percentage'] = request.form.get('financial_cushion_percentage')
    session['risk_preference'] = request.form['risk_preference']
    session['investment_period'] = request.form['investment_period']

    # Retrieve the list of tickers and amounts
    selected_tickers = request.form.getlist('asset_tickers[]')
    investment_amounts = request.form.getlist('investment_amounts[]')

    # Length of tickers and amounts must be the same
    if len(selected_tickers) != len(investment_amounts):
        return jsonify({'error': 'The number of tickers and amounts do not match'}), 400

    # Convert investment amounts to floats and store in a dictionary
    investment_amounts_dict = {ticker: float(amount) for ticker, amount in zip(selected_tickers, investment_amounts)
                               if float(amount) > 0}

    # Check if the total investment is greater than 0
    if sum(investment_amounts_dict.values()) == 0:
        return jsonify({'error': 'The total investment amount cannot be zero'}), 400

    # Store the selected assets and their amounts in the session
    session['selected_assets'] = selected_tickers
    session['investment_amounts'] = investment_amounts_dict

    return redirect(url_for('analyze_portfolio'))


@app.route('/analyze_portfolio', methods=['GET'])
def analyze_portfolio():
    try:
        initial_investment = session.get('initial_investment', '0')
        financial_cushion = session.get('financial_cushion_percentage', '0')
        risk_preference = session.get('risk_preference', 'low')
        investment_period = session.get('investment_period', 'short-term')
        selected_assets = session.get('selected_assets', [])

        # Portfolio preparing for analyze
        started_available_amount = float(initial_investment - (float(initial_investment) * float(financial_cushion) / 100))
        investment_amounts = session.get('investment_amounts', {})
        total_invested_amount = 0
        user_portfolio = []
        investment_percentages = []
        for asset in selected_assets:
            invested_amount = float(float(started_available_amount) * float(investment_amounts.get(asset, 0)) / 100)
            investment_percentages.append(float(investment_amounts.get(asset, 0)))
            total_invested_amount += invested_amount
            user_portfolio.append({'symbol': asset, 'invested_amount': invested_amount})

        # Portfolio risk analyze
        portfolio_vector = prepare_user_portfolio(user_portfolio, asset_risk_classification)
        nn_prediction = predict_portfolio_risk(portfolio_vector, model_FNN, scaler)

        # Portfolio risk comparison with user risk
        risk_comparison, nn_prediction, translated_risk_preference, asset_risks = compare_risk(nn_prediction, risk_preference, user_portfolio, total_invested_amount)

        # Assets similarity analyze
        heatmap_url = analyze_assets_similarity()

        # Volatility chart generation
        volatility_chart_url = generate_volatility_chart_for_portfolio(user_portfolio)

        # Daily return charts generation
        daily_return_charts_url = generate_daily_return_charts(user_portfolio)

        # Pie chart creating
        plot_url = portfolio_pie_chart(user_portfolio)

        # Trends analyzing and assets replacement recomendation
        period_mapping = {'short-term': 7, 'middle-term': 90, 'long-term': 365}
        analysis_period = period_mapping[investment_period]
        translated_investment_period = get_translated_investment_period(investment_period)
        asset_trends = analyze_asset_trends(historical_data, analysis_period)
        replacement_recommendations = generate_replacement_recommendations(user_portfolio, asset_trends, asset_risk_classification)

        data_for_export = collect_portfolio_data_for_export(toFixed(initial_investment, 2), financial_cushion, toFixed(float(initial_investment) * float(financial_cushion) / 100, 2),
                                                            toFixed(float(total_invested_amount), 2), toFixed(float(started_available_amount - total_invested_amount), 2),
                                                            translated_risk_preference, translated_investment_period, nn_prediction, risk_comparison, heatmap_url, plot_url,
                                                            volatility_chart_url, daily_return_charts_url, replacement_recommendations, asset_risks, user_portfolio, investment_percentages)

        export_id = save_data_to_temp_file(data_for_export)

        return render_template('portfolio_analysis.html',
                               initial_investment=toFixed(initial_investment, 2),
                               financial_cushion_percentage=toFixed(float(initial_investment) * float(financial_cushion) / 100, 2),
                               invested_amount=toFixed(float(total_invested_amount), 2),
                               available_amount=toFixed(float(started_available_amount - total_invested_amount), 2),
                               risk_preference=translated_risk_preference,
                               investment_period=translated_investment_period,
                               risk_nn_prediction=nn_prediction,
                               risk_comparison=risk_comparison,
                               heatmap_url=heatmap_url,
                               plot_url=plot_url,
                               volatility_chart_url=volatility_chart_url,
                               daily_return_charts_url=daily_return_charts_url,
                               replacement_recommendations=replacement_recommendations,
                               export_id=export_id)
    except Exception as e:
        app.logger.error('Error: %s', str(e))
        return jsonify({'error': str(e)}), 500


def collect_portfolio_data_for_export(initial_investment, financial_cushion_percentage, financial_cushion, total_invested_amount, available_amount,
                                      translated_risk_preference, translated_investment_period, nn_prediction, risk_comparison, heatmap_chart_url,
                                      pie_chart_url, volatility_chart_url, daily_return_charts_url, replacement_recommendations, asset_risks, user_portfolio, investment_percentages):

    risk_labels_translation = {0: _('low'), 1: _('medium'), 2: _('high')}

    for i, asset in enumerate(user_portfolio):
        asset['invested_percentage'] = investment_percentages[i]
        asset['risk_label'] = risk_labels_translation[asset_risks[i]]

    heatmap_chart_url = base64_to_image(heatmap_chart_url, "/static/Graphs/heatmap_chart.png")
    volatility_chart_url = base64_to_image(volatility_chart_url, "/static/Graphs/volatility_chart.png")
    daily_return_charts_url = base64_to_image(daily_return_charts_url, "/static/Graphs/daily_return_charts.png")
    pie_chart_url = base64_to_image(pie_chart_url, "/static/Graphs/pie_chart.png")

    return {
        "initial_investment": initial_investment,
        "financial_cushion_amount": financial_cushion,
        "financial_cushion_percentage": financial_cushion_percentage,
        "available_amount": available_amount,
        "total_invested_amount": total_invested_amount,
        "risk_preference": translated_risk_preference,
        "investment_period": translated_investment_period,
        "nn_risk_prediction": nn_prediction,
        "risk_comparison": risk_comparison,
        "user_portfolio": user_portfolio,
        "pie_chart": pie_chart_url,
        "heatmap_chart": heatmap_chart_url,
        "volatility_chart": volatility_chart_url,
        "daily_return_charts": daily_return_charts_url,
        "replacement_recommendations": replacement_recommendations
    }


def save_data_to_temp_file(data):
    export_id = str(uuid.uuid4())
    temp_file_path = f"temp_data/{export_id}.json"
    with open(temp_file_path, 'w') as temp_file:
        json.dump(data, temp_file)
    return export_id


@app.route('/download_portfolio/pdf/<export_id>')
def download_pdf(export_id):
    temp_file_path = f"temp_data/{export_id}.json"
    if os.path.exists(temp_file_path):
        with open(temp_file_path, 'r') as temp_file:
            data = json.load(temp_file)
        filename = generate_portfolio_pdf(data)
        return send_file(filename, as_attachment=True)
    else:
        return "Файл не знайдено", 404


@app.route('/download_portfolio/excel/<export_id>')
def download_excel(export_id):
    temp_file_path = f"temp_data/{export_id}.json"
    if os.path.exists(temp_file_path):
        with open(temp_file_path, 'r') as temp_file:
            data = json.load(temp_file)
        filename = generate_portfolio_excel(data)
        return send_file(filename, as_attachment=True)
    else:
        return "Файл не знайдено", 404


def base64_to_image(base64_string, output_filename):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(output_filename)[1])
    base64_string += "=" * ((4 - len(base64_string) % 4) % 4)  # Correct padding if necessary

    try:
        decoded_image = base64.b64decode(base64_string)
        with open(temp_file.name, "wb") as output_file:
            output_file.write(decoded_image)
        return temp_file.name
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def wrap_text(text, width, canvas):
    # Split the text into words
    words = text.split()
    lines = []
    line = ""
    for word in words:
        # Checking the width of the text along with the new word
        if canvas.stringWidth(line + ' ' + word, "DejaVuSans", 10) <= width:
            line += word + " "
        else:
            # If the line is too wide, save it and start a new one
            lines.append(line)
            line = word + " "
    lines.append(line)
    return lines


def generate_portfolio_pdf(data):
    filename = "portfolio_analysis_results.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    max_width = width - 2 * inch
    margin = inch
    y_position = height - margin

    # Title
    c.setFont("DejaVuSans-Bold", 14)
    c.drawString(margin, y_position, _("Investment portfolio analysis"))
    y_position -= 20

    # Basic information
    c.setFont("DejaVuSans", 12)
    basic_info = [
        _("Total funds:") + f" {data['initial_investment']}",
        _("Financial safety cushion:") + f" {data['financial_cushion_amount']} | ({data['financial_cushion_percentage']}%)",
        _("Available funds:") + f" {data['available_amount']}",
        _("Funds in investments:") + f" {data['total_invested_amount']}",
        _("Expected portfolio risk:") + f" {data['risk_preference']}",
        _("The given investment period:") + f" {data['investment_period']}",
        _("Neural network assessment of portfolio risk:") + f" {data['nn_risk_prediction']}",
        _("Recommendation:") + f" {data['risk_comparison']}"
    ]

    for info in basic_info:
        c.drawString(margin, y_position, info)
        y_position -= 15

    # Information about portfolio assets
    y_position -= 10
    c.drawString(margin, y_position, _("Portfolio assets:"))
    y_position -= 15
    c.setFont("DejaVuSans", 10)
    for asset in data['user_portfolio']:
        asset_info = f"{asset['symbol']} - {asset['invested_amount']} | {asset['invested_percentage']}%     (" + _("Risk:") + f" {asset['risk_label']})"
        c.drawString(margin, y_position, asset_info)
        y_position -= 12

    # Asset replacement recommendations
    if data['replacement_recommendations']:
        y_position -= 10
        c.setFont("DejaVuSans-Bold", 12)
        c.drawString(margin, y_position, _("Asset replacement recommendations:"))
        c.setFont("DejaVuSans", 10)
        y_position -= 15
        for asset, details in data['replacement_recommendations'].items():
            rec_info = _("Replace asset:") + f" {asset} (" + _("Current investment:") + f" {details['current_value']}): {', '.join(details['possible_replacements'])}"
            wrapped_text = wrap_text(rec_info, max_width, c)
            for line in wrapped_text:
                c.drawString(margin, y_position, line)
                y_position -= 12

    # Add images
    # Utility function to add images and handle page breaks
    def add_image(path, y_pos, image_height):
        if y_pos - image_height < margin:  # Check if we need a new page
            c.showPage()  # End the current page
            y_pos = height - margin  # Reset y_position for the new page
        c.drawImage(path, margin, y_pos - image_height, width=width - (2 * margin), height=image_height)
        return y_pos - image_height - margin / 2  # Return new y_position

    # Add images with dynamic positioning
    image_heights = {'pie_chart': 350, 'heatmap_chart': 200, 'volatility_chart': 250, 'daily_return_charts': 650}
    for key in ['pie_chart', 'heatmap_chart', 'volatility_chart', 'daily_return_charts']:
        graph_image_path = data[key]
        if graph_image_path:  # Ensure there is an image to add
            y_position = add_image(graph_image_path, y_position, image_heights[key])

    c.save()
    return filename


def generate_portfolio_excel(data):
    filename = "portfolio_analysis_results.xlsx"
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')

    # Basic information
    basic_info_df = pd.DataFrame({
        _("Parameter"): [_("Total funds:"), _("Financial safety cushion:"), _("Financial Cushion Percentage:"),  _("Available funds:"), _("Funds in investments:"), _("Expected portfolio risk:"), _("The given investment period:"), _("Neural network assessment of portfolio risk:"), _("Recommendation:")],
        _("Value"): [data['initial_investment'], data['financial_cushion_amount'], data['financial_cushion_percentage'], data['available_amount'], data['total_invested_amount'], data['risk_preference'], data['investment_period'], data['nn_risk_prediction'], data['risk_comparison']]
    })
    basic_info_df.to_excel(writer, sheet_name=_("Basic information"), index=False)

    # Information about portfolio assets
    assets_df = pd.DataFrame(data['user_portfolio'])
    assets_df.to_excel(writer, sheet_name=_("Portfolio assets"), index=False)

    # Asset replacement recommendations
    replacements = []
    for asset, details in data['replacement_recommendations'].items():
        replacements.append({
            _("Replace asset:"): asset,
            _("Current investment:"): details['current_value'],
            _("Possible replacements:"): ', '.join(details['possible_replacements'])
        })
    if replacements:
        replacements_df = pd.DataFrame(replacements)
        replacements_df.to_excel(writer, sheet_name=_("Asset replacement recommends"), index=False)

    # Adding images to the workbook
    workbook = writer.book
    for key, graph_image_path in data.items():
        if key in ['pie_chart', 'heatmap_chart', 'volatility_chart', 'daily_return_charts'] and graph_image_path:
            worksheet = workbook.add_worksheet(key)  # Each graph gets its own worksheet
            worksheet.insert_image('A1', graph_image_path)  # Adjust position 'A1' as needed

    writer.close()
    return filename


def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"


def calculate_volatility(ohlc_data):
    return ohlc_data['close'].pct_change().rolling(window=21).std() * np.sqrt(252)


def generate_volatility_chart_for_portfolio(user_portfolio):
    plt.figure(figsize=(18, 8))
    for asset in user_portfolio:
        ohlc = historical_data_OHLC.get(asset['symbol'])
        if ohlc is not None:
            volatility = calculate_volatility(ohlc)
            plt.plot(volatility, label=asset['symbol'])

    plt.legend()
    plt.title(_('Volatility Assets Chart'))
    plt.xlabel(_('Date'))
    plt.ylabel(_('Volatility'))

    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode('utf8')
    return chart_url


def generate_daily_return_charts(user_portfolio):
    num_assets = len(user_portfolio)
    num_rows = (num_assets + 1) // 2

    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(14, num_rows * 4))
    axes = axes.flatten()

    for i, asset in enumerate(user_portfolio):
        ticker = asset['symbol']
        ohlc = historical_data_OHLC.get(ticker)
        if ohlc is not None:
            daily_returns = ohlc['close'].pct_change()
            axes[i].plot(daily_returns, label=f'{ticker} ' + _('Daily Returns'), marker='d')
            axes[i].legend()
            axes[i].set_xlabel(_('Date'))
            axes[i].set_ylabel(_('Daily Returns'))

    if num_assets % 2 != 0:
        fig.delaxes(axes[-1])

    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    charts_url = base64.b64encode(img.getvalue()).decode('utf8')
    return charts_url


@app.route('/get_asset_chart', methods=['POST'])
def get_asset_chart():
    data = request.get_json()

    if not data or 'ticker' not in data or 'period' not in data:
        return jsonify({'error': 'Missing ticker or period in the request'}), 400

    ticker = data['ticker']
    period = int(data['period'])
    ohlc = historical_data_OHLC.get(ticker)

    if ohlc is None or ohlc.empty:
        return jsonify({'error': f'No data available for ticker: {ticker}'}), 404

    if len(ohlc) < period:
        return jsonify({'error': f'Not enough data to plot for the period: {period}'}), 400

    selected_ohlc = ohlc[-period:]

    mpf_style = mpf.make_mpf_style(base_mpf_style='charles',
                                   marketcolors=mpf.make_marketcolors(up='#009900', down='#d21427', inherit=True))
    colors = ['#009900' if row['close'] >= row['open'] else '#d21427' for index, row in selected_ohlc.iterrows()]

    addplot_volume = mpf.make_addplot(selected_ohlc['volume'], panel=1, type='bar', color=colors, ylabel=_("Trading\nVolume"))
    addplot_volume['yaxis_formatter'] = mticker.FormatStrFormatter('%d')

    img = BytesIO()
    mpf.plot(selected_ohlc, type='candle', addplot=addplot_volume, style=mpf_style, ylabel=_("Price ($)"),
             title=_("OHLC Candlestick Chart for") + f" {ticker}", datetime_format='%Y-%m-%d',
             figsize=(13.5, 6), savefig=dict(fname=img, dpi=100, bbox_inches='tight'))

    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return jsonify({'chart_url': f"data:image/png;base64,{plot_url}"})


@app.route('/search_assets/<query>')
def search_assets(query):
    matching_assets = [ticker for ticker in all_tickers if query.upper() in ticker]
    return jsonify(matching_assets)


@app.route('/check_ticker/<ticker>')
def check_ticker(ticker):
    if ticker in all_tickers:
        return jsonify({'exists': True})
    else:
        return jsonify({'exists': False})


def analyze_trend_sector(data, period):
    if len(data) < period:
        print(f"Insufficient data for analysis")
        return 'insufficient data'

    last_price = data[-1]
    mean_price = sum(data[-period:]) / period
    return 'green' if last_price >= mean_price else 'red'


def analyze_asset_trends(historical_data, analysis_period):
    asset_trends = {}
    for ticker, data in historical_data.items():
        trend = analyze_trend_sector(data, analysis_period)
        asset_trends[ticker] = trend
    return asset_trends


def generate_replacement_recommendations(user_portfolio, asset_trends, asset_risk_classification):
    recommendations = {}
    for asset in user_portfolio:
        if asset_trends[asset['symbol']] == 'red':  # If the asset is declining
            # Identify the sector of the declining asset
            risk_level, asset_type, sector_name = classify_asset_sector(asset['symbol'], asset_risk_classification)

            # Get all assets within the same sector
            sector_assets = asset_risk_classification[risk_level][asset_type][sector_name]
            # Filter for assets that are in the same sector and are rising
            sector_rising_assets = [ticker for ticker in sector_assets if asset_trends.get(ticker) == 'green']
            # Exclude the declining asset from its own replacements
            sector_rising_assets = [ticker for ticker in sector_rising_assets if ticker != asset['symbol']]

            if sector_rising_assets:
                recommendations[asset['symbol']] = {
                    'current_value': asset['invested_amount'],
                    'possible_replacements': sector_rising_assets
                }
    return recommendations


def classify_asset_sector(symbol, asset_risk_classification):
    for risk_level, asset_types in asset_risk_classification.items():
        for asset_type, sectors in asset_types.items():
            for sector_name, assets in sectors.items():
                if symbol in assets:
                    return risk_level, asset_type, sector_name
    return None


def classify_asset_risk(symbol, asset_risk_classification):
    for risk_level, categories in asset_risk_classification.items():
        for category, assets in categories.items():
            for asset_type, symbols in assets.items():
                if symbol in symbols:
                    return {'low_risk': 0, 'medium_risk': 1, 'high_risk': 2}[risk_level]
    return None


def prepare_user_portfolio(user_portfolio, asset_risk_classification):
    # Initialize a default portfolio vector with zeros
    default_portfolio_vector = [0] * NUM_ASSETS * FEATURES_PER_ASSET

    total_investment = sum(asset['invested_amount'] for asset in user_portfolio if asset['invested_amount'] > 0)

    if total_investment == 0:
        raise ValueError('Загальна сума інвестицій не може бути нульовою')

    # Populate the portfolio vector with the user's assets
    for i, asset in enumerate(user_portfolio):
        if i >= NUM_ASSETS:
            break  # Ensure we only consider the number of assets the model can handle
        risk_level = classify_asset_risk(asset['symbol'], asset_risk_classification)
        if risk_level is not None and asset['invested_amount'] > 0:
            weight = asset['invested_amount'] / total_investment
            weighted_risk = risk_level * weight
            # Each asset takes up two slots in the vector: one for weighted_risk and one for weight
            default_portfolio_vector[i * FEATURES_PER_ASSET] = weighted_risk
            default_portfolio_vector[i * FEATURES_PER_ASSET + 1] = weight

    return default_portfolio_vector


def predict_portfolio_risk(portfolio, model, scaler):
    X_new = np.array(portfolio).reshape(1, -1)
    X_new_scaled = scaler.transform(X_new)
    prediction = model.predict(X_new_scaled)
    return prediction[0][0]


risk_preferences_translations = {
            'low': _('low'),
            'medium': _('medium'),
            'high': _('high')
        }


investment_period_translations = {
            'short-term': _('short-term'),
            'middle-term': _('middle-term'),
            'long-term': _('long-term')
        }


def get_translated_risk_preference(risk_preference):
    return str(_l(risk_preferences_translations.get(risk_preference, 'low')))


def get_translated_investment_period(investment_period):
    return str(_l(investment_period_translations.get(investment_period, 'short-term')))


def compare_risk(predicted_risk, user_risk_preference, user_portfolio, total_invested_amount):
    risk_levels = {'low': 0, 'medium': 1, 'high': 2}
    user_risk_level = risk_levels.get(user_risk_preference.lower(), 0)
    translated_risk_preference = get_translated_risk_preference(user_risk_preference)
    risk_weights = {'low_risk': 0, 'medium_risk': 0, 'high_risk': 0}
    asset_risks = []

    for asset in user_portfolio:
        risk_level = classify_asset_risk(asset['symbol'], asset_risk_classification)
        asset_risks.append(risk_level)
        risk_weights['low_risk' if risk_level == 0 else 'medium_risk' if risk_level == 1 else 'high_risk'] += asset['invested_amount'] / total_invested_amount

    risk_score = (risk_weights['low_risk'] * 1 + risk_weights['medium_risk'] * 2 + risk_weights['high_risk'] * 3) / 3

    predicted_risk = float((predicted_risk + risk_score) / 2)

    if predicted_risk <= 0.33:
        predicted_risk_level = 0  # low
    elif predicted_risk <= 0.66:
        predicted_risk_level = 1  # medium
    else:
        predicted_risk_level = 2  # high

    if predicted_risk_level == user_risk_level:
        return _("Portfolio risk corresponds to desired risk ") + f" ({translated_risk_preference}).", predicted_risk, translated_risk_preference, asset_risks
    elif predicted_risk_level < user_risk_level:
        return _("Portfolio risk is lower than desired (") + translated_risk_preference + _("). Perhaps it is worth increasing the riskiness of the portfolio."), predicted_risk, translated_risk_preference, asset_risks
    else:
        return _("Portfolio risk is higher than desired (") + translated_risk_preference + _("). We recommend reviewing the composition of the portfolio to reduce risk."), predicted_risk, translated_risk_preference, asset_risks


@app.route('/portfolio_pie_chart')
def portfolio_pie_chart(user_portfolio):
    pie_chart_data = prepare_pie_chart_data(user_portfolio, asset_risk_classification)
    translated_risk = _("Risk:")

    # Low risk: Cold colors
    low_risk_colors = [
        '#ADD8E6',  # Світло-синій
        '#87CEEB',  # Блакитний
        '#0F52BA',  # Сапфіровий
        '#00008B',  # Глибокий синій
        '#48D1CC',  # Морської хвилі
        '#90EE90',  # Світло-зелений
        '#32CD32',  # Лаймовий
        '#008000',  # Зелений
        '#006400',  # Темно-зелений
        '#50C878'  # Смарагдовий
    ]
    # Medium risk: Warm colors
    medium_risk_colors = [
        '#FFFFE0',  # Світло-жовтий
        '#FFFACD',  # Лимонний
        '#FFD700',  # Канарковий
        '#FFD700',  # Золотистий
        '#FDBCB4',  # Пастельно-оранжевий
        '#FFA500',  # Оранжевий
        '#FF8C00',  # Темно-оранжевий
        '#FF7F50',  # Кораловий
        '#E2725B',  # Теракотовий
        '#B87333'  # Мідний
    ]
    # High risk: Hot colors
    high_risk_colors = [
        '#FF6A6A',  # Світло-червоний
        '#FF0000',  # Червоний
        '#DC143C',  # Карміновий
        '#800000',  # Бордовий
        '#C71585',  # Вишневий
        '#E6E6FA',  # Світло-фіолетовий
        '#E6E6FA',  # Лавандовий (той же, що і світло-фіолетовий)
        '#8A2BE2',  # Фіолетовий
        '#9400D3',  # Темно-фіолетовий
        '#9966CC'  # Аметистовий
    ]
    risk_colors = {
        0: low_risk_colors,  # low risk
        1: medium_risk_colors,  # medium risk
        2: high_risk_colors  # high risk
    }

    fig, ax = plt.subplots()
    sizes = [data['invested_amount'] for data in pie_chart_data]
    labels = [f"{data['symbol']} ({translated_risk} {data['risk_level']})" for data in pie_chart_data]
    colors = [risk_colors[data['risk_level']][index % 10] for index, data in enumerate(pie_chart_data)]

    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    for autotext in autotexts:
        autotext.set_path_effects([PathEffects.withStroke(linewidth=3, foreground="white")])

    ax.axis('equal')

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return plot_url


def prepare_pie_chart_data(user_portfolio, asset_risk_classification):
    pie_chart_data = []
    for asset in user_portfolio:
        symbol = asset['symbol']
        invested_amount = asset['invested_amount']
        risk_level = classify_asset_risk(symbol, asset_risk_classification)
        if risk_level is not None:
            pie_chart_data.append({
                'symbol': symbol,
                'invested_amount': invested_amount,
                'risk_level': risk_level
            })
    return pie_chart_data


def calculate_percentage_change(data):
    return (data[1:] - data[:-1]) / data[:-1]


def z_score_normalization(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


def rank_transform(matrix):
    flat = matrix.flatten()
    ranks = flat.argsort().argsort().astype(float)
    ranks = ranks.reshape(matrix.shape)
    ranks = 2 * ((ranks - np.min(ranks)) / (np.max(ranks) - np.min(ranks))) - 1
    return ranks


@app.route('/analyze_assets_similarity', methods=['GET'])
def analyze_assets_similarity():
    user_portfolio = session.get('selected_assets', [])

    if not user_portfolio:
        return jsonify({'error': 'Portfolio is required'}), 400

    num_assets = len(user_portfolio)
    similarity_matrix = np.zeros((num_assets, num_assets))
    pearson_matrix = np.zeros((num_assets, num_assets))

    left_assets_data = []
    right_assets_data = []

    for i in range(num_assets):
        for j in range(i + 1, num_assets):
            data_i = z_score_normalization(calculate_percentage_change(loaded_data[user_portfolio[i]]['close'][:MIN_LENGTH + 1].values))
            data_j = z_score_normalization(calculate_percentage_change(loaded_data[user_portfolio[j]]['close'][:MIN_LENGTH + 1].values))
            left_assets_data.append(data_i)
            right_assets_data.append(data_j)

    left_assets_data = np.array(left_assets_data)
    right_assets_data = np.array(right_assets_data)

    predictions = prediction_model_SNN.predict([left_assets_data, right_assets_data])

    k = 0
    for i in range(num_assets):
        for j in range(i + 1, num_assets):
            similarity_score = 1 - predictions[k]
            similarity_matrix[i][j] = similarity_matrix[j][i] = similarity_score
            pearson_corr = np.corrcoef(loaded_data[user_portfolio[i]]['close'][:MIN_LENGTH + 1].values,
                                       loaded_data[user_portfolio[j]]['close'][:MIN_LENGTH + 1].values)[0, 1]
            pearson_matrix[i][j] = pearson_matrix[j][i] = pearson_corr
            k += 1

    np.fill_diagonal(similarity_matrix, 1)
    np.fill_diagonal(pearson_matrix, 1)
    transformed_data = rank_transform(similarity_matrix)
    np.fill_diagonal(transformed_data, 1)
    print(f"NN original corr matrix:\n{similarity_matrix}\n\n\n")
    print(f"NN transformed corr matrix:\n{transformed_data}\n\n\n")
    print(f"Pearson corr matrix:\n{pearson_matrix}")

    cmap_continuous = sns.color_palette("summer", as_cmap=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(transformed_data, cmap=cmap_continuous, square=True, cbar=True, ax=ax1, xticklabels=user_portfolio, yticklabels=user_portfolio)
    ax1.title.set_text(_('Neural Network Similarity Heatmap'))
    sns.heatmap(pearson_matrix, cmap=cmap_continuous, square=True, cbar=True, ax=ax2, xticklabels=user_portfolio, yticklabels=user_portfolio)
    ax2.title.set_text(_('Pearson Correlation Heatmap'))
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    img.seek(0)
    heatmap_url = base64.b64encode(img.getvalue()).decode('utf8')

    return heatmap_url


def save_or_encode_plot(plt, save_file=False, file_path="plot.png"):
    if save_file:
        plt.savefig(file_path)
        return file_path
    else:
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        img_base64 = base64.b64encode(img.read()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"


if __name__ == '__main__':
    app.run(debug=True)
