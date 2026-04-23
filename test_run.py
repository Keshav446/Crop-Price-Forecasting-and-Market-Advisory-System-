import traceback
with open("test_run.log", "w", encoding="utf-8") as out:
    try:
        with open("crop_price_forecasting.py", encoding="utf-8") as f:
            code = f.read()
        exec(code)
        out.write("Success")
    except Exception as e:
        out.write(traceback.format_exc())
