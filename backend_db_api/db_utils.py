# db_utils.py
import pandas as pd
import psycopg2
import json

from datetime import datetime

def insert_prediction(prediction_data):
    conn = None
    cur = None
    try:
        conn = psycopg2.connect(**prediction_data["conn_params"])
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO user_predictions (
                timestamp, model_name, drawn_digit, predicted_digit,
                probability, probabilities, correct, background_color, pen_color
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                datetime.now(),
                prediction_data["model_name"],
                prediction_data["drawn_digit"],
                prediction_data["predicted_digit"],
                prediction_data["probability"],
                prediction_data["probabilities"],
                prediction_data["correct"],
                prediction_data.get("background_color"),
                prediction_data.get("pen_color"),
            )
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"Error inserting prediction: {e}")
        return False
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

def get_predictions(conn_params):
    try:
        conn = psycopg2.connect(**conn_params)
        df = pd.read_sql("SELECT * FROM user_predictions", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error fetching predictions: {e}")
        return False