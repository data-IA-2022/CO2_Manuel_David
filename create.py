from utils import get_df_from_db, get_engine
from ydata_profiling import ProfileReport
import dotenv


def main():
    dotenv.load_dotenv()
    # create report.html
    engine = get_engine(echo_arg=True)

    df = get_df_from_db(engine)

    profile = ProfileReport(df, title="Profiling Report")
    profile.to_file("templates/report.html")
    print('report.html has been generated')
   


if __name__=='__main__':
    main()