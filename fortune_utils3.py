import math
import datetime
from datetime import timedelta, timezone
from korean_lunar_calendar import KoreanLunarCalendar
from decimal import Decimal, getcontext

# (옵션) Decimal 정밀도 설정
getcontext().prec = 50

# -----------------------------------------------
# 1) 간지(干支) 기초 데이터
# -----------------------------------------------
HEAVENLY_STEMS = ["갑", "을", "병", "정", "무", "기", "경", "신", "임", "계"]
EARTHLY_BRANCHES = ["자", "축", "인", "묘", "진", "사", "오", "미", "신", "유", "술", "해"]
GANZHI_60 = [(HEAVENLY_STEMS[i % 10], EARTHLY_BRANCHES[i % 12]) for i in range(60)]

# (생략 없이) 십성, 오행, 12운성 테이블은 그대로 사용
TEN_GODS_STEM_TABLE = {
    "갑": ["비견", "겁재", "식신", "상관", "편재", "정재", "편관", "정관", "편인", "정인"],
    "을": ["겁재", "비견", "상관", "식신", "정재", "편재", "정관", "편관", "정인", "편인"],
    "병": ["편인", "정인", "비견", "겁재", "식신", "상관", "편재", "정재", "편관", "정관"],
    "정": ["정인", "편인", "겁재", "비견", "상관", "식신", "정재", "편재", "정관", "편관"],
    "무": ["편관", "정관", "편인", "정인", "비견", "겁재", "식신", "상관", "편재", "정재"],
    "기": ["정관", "편관", "정인", "편인", "겁재", "비견", "상관", "식신", "정재", "편재"],
    "경": ["편재", "정재", "편관", "정관", "편인", "정인", "비견", "겁재", "식신", "상관"],
    "신": ["정재", "편재", "정관", "편관", "정인", "편인", "겁재", "비견", "상관", "식신"],
    "임": ["식신", "상관", "편재", "정재", "편관", "정관", "편인", "정인", "비견", "겁재"],
    "계": ["상관", "식신", "정재", "편재", "정관", "편관", "정인", "편인", "겁재", "비견"]
}
TEN_GODS_BRANCH_TABLE = {
    "갑": ["정인", "정재", "비견", "겁재", "편재", "식신", "상관", "정관", "편관", "편인", "정재", "편인"],
    "을": ["편인", "편재", "겁재", "비견", "정재", "상관", "식신", "편관", "정관", "정인", "편재", "정인"],
    "병": ["정관", "정재", "편인", "정인", "비견", "겁재", "식신", "상관", "편재", "편관", "정재", "편관"],
    "정": ["편관", "편재", "정인", "정인", "겁재", "비견", "상관", "식신", "정재", "정관", "편재", "정관"],
    "무": ["정재", "정재", "편관", "정관", "비견", "겁재", "식신", "상관", "편재", "정재", "편관", "정재"],
    "기": ["편재", "편재", "정관", "편관", "겁재", "비견", "상관", "식신", "정재", "편재", "정관", "편재"],
    "경": ["식신", "정재", "편재", "정재", "편인", "정인", "비견", "겁재", "식신", "상관", "정재", "상관"],
    "신": ["상관", "편재", "정재", "편재", "정인", "편인", "겁재", "비견", "상관", "식신", "편재", "식신"],
    "임": ["편재", "식신", "상관", "식신", "편관", "정관", "편인", "정인", "비견", "겁재", "식신", "겁재"],
    "계": ["식신", "상관", "편재", "상관", "정관", "편관", "정인", "편인", "겁재", "비견", "상관", "비견"]
}
ELEMENTS = {
    "갑": "목", "을": "목",
    "병": "화", "정": "화",
    "무": "토", "기": "토",
    "경": "금", "신": "금",
    "임": "수", "계": "수",
    "자": "수", "축": "토",
    "인": "목", "묘": "목",
    "진": "토", "사": "화",
    "오": "화", "미": "토",
    "신": "금", "유": "금",
    "술": "토", "해": "수"
}
TWELVE_STATES_TABLE = {
    "갑": ["목욕", "관대", "건록", "제왕", "쇠", "병", "사", "묘", "절", "태", "양", "장생생"],
    "을": ["병", "쇠", "제왕", "건록", "관대", "목욕", "장생", "양", "태", "절", "묘", "사"],
    "병": ["태", "양", "장생", "목욕", "관대", "건록", "제왕", "쇠", "병", "사", "묘", "절"],
    "정": ["절", "묘", "사", "병", "쇠", "제왕", "건록", "관대", "목욕", "장생", "양", "태"],
    "무": ["태", "양", "장생", "목욕", "관대", "건록", "제왕", "쇠", "병", "사", "묘", "절"],
    "기": ["절", "묘", "사", "병", "쇠", "제왕", "건록", "관대", "목욕", "장생", "양", "태"],
    "경": ["사", "묘", "절", "태", "양", "장생", "목욕", "관대", "건록", "제왕", "쇠", "병"],
    "신": ["장생", "양", "태", "절", "묘", "사", "병", "쇠", "제왕", "건록", "관대", "목욕"],
    "임": ["제왕", "쇠", "병", "사", "묘", "절", "태", "양", "장생", "목욕", "관대", "건록"],
    "계": ["건록", "관대", "목욕", "장생", "양", "태", "절", "묘", "사", "병", "쇠", "제왕"]
}

# 기준일 (1936-02-12, 갑자일)
REFERENCE_DATE = datetime.datetime(1936, 2, 12, 0, 0, 0)

def day_index_from_reference(dt: datetime.datetime) -> int:
    """REFERENCE_DATE부터 dt까지의 일수 차이"""
    return (dt.date() - REFERENCE_DATE.date()).days

# -----------------------------------------------
# 2) 천문 계산 (내장 모듈만 사용, ΔT 보정 및 뉴턴-랩슨 미세조정 포함)
# -----------------------------------------------
def delta_t_for_dt(dt: datetime.datetime) -> float:
    """
    UT datetime에 대해 ΔT(초)를 추정 (2000~2100년 범위)
    ΔT = 62.92 + 0.32217*(y-2000) + 0.005589*(y-2000)^2
    """
    y = dt.year + (dt.month - 0.5) / 12.0
    return 62.92 + 0.32217 * (y - 2000) + 0.005589 * (y - 2000)**2

def julday_from_utc(dt: datetime.datetime) -> float:
    """UTC datetime을 Julian Day (JD)로 변환"""
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    if month <= 2:
        year -= 1
        month += 12
    A = math.floor(year / 100)
    B = 2 - A + math.floor(A / 4)
    JD = (math.floor(365.25 * (year + 4716)) +
          math.floor(30.6001 * (month + 1)) +
          day + B - 1524.5 + hour / 24.0)
    return JD

def datetime_from_julday(jd: float, tz=0) -> datetime.datetime:
    """Julian Day와 시차(tz)를 이용해 로컬 datetime 반환"""
    J = jd + 0.5
    Z = int(J)
    F = J - Z
    if Z < 2299161:
        A = Z
    else:
        alpha = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + alpha - int(alpha / 4)
    B = A + 1524
    C = int((B - 122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B - D) / 30.6001)
    day = B - D - int(30.6001 * E) + F
    if E < 14:
        month = E - 1
    else:
        month = E - 13
    if month > 2:
        year = C - 4716
    else:
        year = C - 4715
    day_int = int(day)
    day_fraction = day - day_int
    hours = day_fraction * 24
    hour_int = int(hours)
    minutes = (hours - hour_int) * 60
    minute_int = int(minutes)
    seconds = (minutes - minute_int) * 60
    second_int = int(round(seconds))
    if second_int >= 60:
        second_int -= 60
        minute_int += 1
    if minute_int >= 60:
        minute_int -= 60
        hour_int += 1
    dt_utc = datetime.datetime(year, month, day_int, hour_int, minute_int, second_int, tzinfo=datetime.timezone.utc)
    return dt_utc + datetime.timedelta(hours=tz)

def sun_ecliptic_longitude(jd: float) -> float:
    """
    Meeus 공식 (고차항 보정 포함)과 ΔT 보정을 적용하여,
    주어진 JD(UT)를 TT로 보정한 후 태양의 보정 황경(apparent longitude)을 계산.
    """
    dt_utc = datetime_from_julday(jd, tz=0)
    deltaT = delta_t_for_dt(dt_utc)  # 초 단위
    jd_tt = jd + deltaT / 86400.0
    T = (jd_tt - 2451545.0) / 36525.0
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T**2
    L0 %= 360.0
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T**2
    M %= 360.0
    M_rad = math.radians(M)
    C = ((1.914602 - 0.004817 * T - 0.000014 * T**2) * math.sin(M_rad) +
         (0.019993 - 0.000101 * T) * math.sin(2 * M_rad) +
         0.000289 * math.sin(3 * M_rad))
    true_long = (L0 + C) % 360.0
    Omega = 125.04 - 1934.136 * T
    apparent_long = (true_long - 0.00569 - 0.00478 * math.sin(math.radians(Omega))) % 360.0
    return apparent_long

def angle_diff(lon: float, target_deg: float) -> float:
    """lon과 target_deg의 차이를 -180 ~ +180 범위로 조정"""
    d = (lon - target_deg) % 360.0
    if d > 180:
        d -= 360
    return d

def newton_refine(target_deg: float, jd_initial: float, iterations=10, h=1e-6) -> float:
    """
    뉴턴-랩슨 방법으로 target_deg에 해당하는 JD를 미세 조정.
    f(jd) = angle_diff(sun_ecliptic_longitude(jd), target_deg)
    """
    jd = jd_initial
    for _ in range(iterations):
        f_val = angle_diff(sun_ecliptic_longitude(jd), target_deg)
        f_plus = angle_diff(sun_ecliptic_longitude(jd + h), target_deg)
        f_minus = angle_diff(sun_ecliptic_longitude(jd - h), target_deg)
        derivative = (f_plus - f_minus) / (2 * h)
        if derivative == 0:
            break
        jd -= f_val / derivative
    return jd

def find_solar_longitude_cross(target_deg: float, jd_start: float, jd_end: float, max_iter=50) -> float:
    """
    jd_start ~ jd_end 범위 내에서 태양 황경이 target_deg에 도달하는 JD를
    이분법과 뉴턴-랩슨 방법으로 미세 조정하여 찾습니다.
    """
    v_start = angle_diff(sun_ecliptic_longitude(jd_start), target_deg)
    v_end = angle_diff(sun_ecliptic_longitude(jd_end), target_deg)
    if (v_start > 0 and v_end > 0) or (v_start < 0 and v_end < 0):
        return find_solar_longitude_cross(target_deg, jd_start - 30, jd_end + 30, max_iter)
    tol = 1e-10
    for _ in range(max_iter):
        jd_mid = (jd_start + jd_end) / 2.0
        v_mid = angle_diff(sun_ecliptic_longitude(jd_mid), target_deg)
        if abs(v_mid) < tol:
            jd_mid = newton_refine(target_deg, jd_mid)
            return jd_mid
        if (v_start > 0 and v_mid > 0) or (v_start < 0 and v_mid < 0):
            jd_start = jd_mid
            v_start = v_mid
        else:
            jd_end = jd_mid
            v_end = v_mid
    jd_mid = (jd_start + jd_end) / 2.0
    return newton_refine(target_deg, jd_mid)

# -----------------------------------------------
# 3) 월주 계산 – 태양황경을 직접 이용하는 방식
#
# 전통 공식:
#   solar_month = floor((apparent_longitude + 15) / 30) mod 12 + 1
#   월지는 고정 순서: ["묘", "진", "사", "오", "미", "신", "유", "술", "해", "자", "축", "인"]
#   월간은 (연간 천간 인덱스 × 2 + solar_month + 2) mod 10
# -----------------------------------------------
def calculate_month_pillar(dt_local: datetime.datetime, year_gan: str) -> (str, str):
    """
    dt_local의 태양 보정 황경을 이용하여 solar_month를 계산한 후,
    월간과 월지를 산출합니다.
    """
    # dt_local의 UTC JD 구하기
    jd = julday_from_utc(dt_local.astimezone(datetime.timezone.utc))
    lon = sun_ecliptic_longitude(jd)
    solar_month = int(math.floor((lon + 15) / 30)) % 12 + 1
    # 전통적 사주에서 월지 고정 순서 (1월->묘, 2월->진, …, 12월->인)
    month_branch_order = ["묘", "진", "사", "오", "미", "신", "유", "술", "해", "자", "축", "인"]
    month_branch = month_branch_order[solar_month - 1]
    year_stem_index = HEAVENLY_STEMS.index(year_gan)
    month_stem_index = (year_stem_index * 2 + solar_month + 2) % 10
    month_stem = HEAVENLY_STEMS[month_stem_index]
    return month_stem, month_branch

# -----------------------------------------------
# 4) 사주 계산 (년주, 월주, 일주, 시주)
#
# - 연주: 입춘 기준 (입춘 이후이면 당해, 아니면 전년도)
# - 월주: calculate_month_pillar()로 계산
# - 일주: 기준일(1936-02-12, 갑자일)부터의 일수
# - 시주: 2시간 단위 (23시 이후는 익일로 처리)
# -----------------------------------------------
def get_year_ganzhi(year: int, after_ipchun: bool) -> (str, str):
    """1864년 기준 갑자년을 사용하여 연주 계산"""
    base_year = 1864
    if not after_ipchun:
        year -= 1
    offset = (year - base_year) % 60
    return GANZHI_60[offset]

def get_hour_ganzhi(day_gz_index: int, hour_0_23: int) -> (str, str):
    """
    시주는 (일간 인덱스 × 2 + 시간대) mod 10,
    시간대는 2시간 단위 (예: 0~1시→0, 2~3시→1, …)
    """
    hour_zhi_index = (hour_0_23 // 2) % 12
    day_stem_index = day_gz_index % 10
    hour_stem_index = (day_stem_index * 2 + hour_zhi_index) % 10
    return (HEAVENLY_STEMS[hour_stem_index], EARTHLY_BRANCHES[hour_zhi_index])

def calculate_day_pillar(dt_local: datetime.datetime) -> (str, str):
    """
    기준일(1936-02-12, 갑자일)부터의 일수로 일주(일간, 일지)를 결정.
    (23시 이후는 익일로 처리)
    """
    if dt_local.hour >= 23:
        dt_local += timedelta(days=1)
        dt_local = dt_local.replace(hour=dt_local.hour - 24)
    day_diff = (dt_local.date() - REFERENCE_DATE.date()).days
    return GANZHI_60[day_diff % 60]

def calculate_twelve_state(day_stem: str, branch: str) -> str:
    """
    주어진 일간과 지지로부터 12운성을 결정.
    """
    if day_stem not in TWELVE_STATES_TABLE:
        raise ValueError(f"일간 '{day_stem}'이 올바르지 않습니다.")
    states = TWELVE_STATES_TABLE[day_stem]
    if branch not in EARTHLY_BRANCHES:
        raise ValueError(f"지지 '{branch}'이 올바르지 않습니다.")
    branch_index = EARTHLY_BRANCHES.index(branch)
    return states[branch_index]

def get_saju(year: int, month: int, day: int, hour: int = 0, minute: int = 0,
             is_lunar: bool = False, is_leap_month: bool = False, tz: int = 9) -> dict:
    # (A) 음력 입력이면 양력 변환 (korean_lunar_calendar 사용)
    if is_lunar:
        cal = KoreanLunarCalendar()
        cal.setLunarDate(year, month, day, is_leap_month)
        solar_y = cal.getSolarYear()
        solar_m = cal.getSolarMonth()
        solar_d = cal.getSolarDay()
    else:
        solar_y, solar_m, solar_d = year, month, day

    # (B) 로컬 datetime 생성 (예: KST=UTC+9)
    kst = timezone(timedelta(hours=tz))
    dt_local = datetime.datetime(solar_y, solar_m, solar_d, hour, minute, tzinfo=kst)

    # (C) 입춘 기준 (여기서는 기본값 사용)
    ipchun_dt = datetime.datetime(dt_local.year, 2, 4, 5, 0, tzinfo=kst)

    # (D) 연주 결정: 입춘 이후이면 당해, 아니면 전년도
    after_ipchun = dt_local >= ipchun_dt
    year_gan, year_zhi = get_year_ganzhi(dt_local.year, after_ipchun)

    # (E) 월주 결정: calculate_month_pillar()를 이용
    month_gan, month_zhi = calculate_month_pillar(dt_local, year_gan)

    # (F) 일주 결정
    day_gan, day_zhi = calculate_day_pillar(dt_local)

    # (G) 시주 결정
    day_index = day_index_from_reference(dt_local) % 60
    hour_gan, hour_zhi = get_hour_ganzhi(day_index, dt_local.hour)

    return {
        "year": (year_gan, year_zhi),
        "year_element": (ELEMENTS.get(year_gan, ""), ELEMENTS.get(year_zhi, "")),
        "month": (month_gan, month_zhi),
        "month_element": (ELEMENTS.get(month_gan, ""), ELEMENTS.get(month_zhi, "")),
        "day": (day_gan, day_zhi),
        "day_element": (ELEMENTS.get(day_gan, ""), ELEMENTS.get(day_zhi, "")),
        "hour": (hour_gan, hour_zhi),
        "hour_element": (ELEMENTS.get(hour_gan, ""), ELEMENTS.get(hour_zhi, "")),
        "calc_info": {
            "final_solar_date": dt_local.strftime("%Y-%m-%d %H:%M"),
            "ipchun_date": ipchun_dt.strftime("%Y-%m-%d %H:%M"),
            "after_ipchun": after_ipchun,
            "computed_month": calculate_month_pillar(dt_local, year_gan),
            "is_lunar_input": is_lunar,
            "is_leap_month": is_leap_month,
        }
    }

def get_ten_god_stem(day_stem: str, other_stem: str) -> str:
    """일간과 다른 천간 간의 십성 반환"""
    return TEN_GODS_STEM_TABLE[day_stem][HEAVENLY_STEMS.index(other_stem)]

def get_ten_god_branch(day_stem: str, branch: str) -> str:
    """일간과 지지 간의 십성 반환"""
    return TEN_GODS_BRANCH_TABLE[day_stem][EARTHLY_BRANCHES.index(branch)]

def get_ten_god(saju: dict) -> dict:
    day_stem = saju["day"][0]
    return {
        "year_ten_god_stem": get_ten_god_stem(day_stem, saju["year"][0]),
        "year_ten_god_branch": get_ten_god_branch(day_stem, saju["year"][1]),
        "month_ten_god_stem": get_ten_god_stem(day_stem, saju["month"][0]),
        "month_ten_god_branch": get_ten_god_branch(day_stem, saju["month"][1]),
        "day_ten_god_stem": "일원",
        "day_ten_god_branch": get_ten_god_branch(day_stem, saju["day"][1]),
        "hour_ten_god_stem": get_ten_god_stem(day_stem, saju["hour"][0]),
        "hour_ten_god_branch": get_ten_god_branch(day_stem, saju["hour"][1])
    }

def get_twelve_state(saju: dict) -> dict:
    day_stem = saju["day"][0]
    return {
        "year_twelve_state": calculate_twelve_state(day_stem, saju["year"][1]),
        "month_twelve_state": calculate_twelve_state(day_stem, saju["month"][1]),
        "day_twelve_state": calculate_twelve_state(day_stem, saju["day"][1]),
        "hour_twelve_state": calculate_twelve_state(day_stem, saju["hour"][1])
    }

# -----------------------------------------------
# 6) 테스트 예시
# -----------------------------------------------
if __name__ == "__main__":
    # 1997년 3월 24일 17:00 KST → 기대 결과: 월주가 "계묘"여야 함.
    saju1 = get_saju(1997, 3, 24, 17, 0, is_lunar=False, tz=9)
    print("=== 양력 1997-03-24 17:00 KST ===")
    print("연주:", "".join(saju1["year"]))
    print("연주 오행:", "".join(saju1["year_element"]))
    print("월주:", "".join(saju1["month"]))
    print("월주 오행:", "".join(saju1["month_element"]))
    print("일주:", "".join(saju1["day"]))
    print("일주 오행:", "".join(saju1["day_element"]))
    print("시주:", "".join(saju1["hour"]))
    print("시주 오행:", "".join(saju1["hour_element"]))
    print("calc_info:", saju1["calc_info"])

    ten_god = get_ten_god(saju1)
    print("일간:", saju1["day"][0])
    print("연주 천간 십성:", ten_god["year_ten_god_stem"])
    print("연주 지지 십성:", ten_god["year_ten_god_branch"])
    print("월주 천간 십성:", ten_god["month_ten_god_stem"])
    print("월주 지지 십성:", ten_god["month_ten_god_branch"])
    print("일주 천간 십성:", ten_god["day_ten_god_stem"])
    print("일주 지지 십성:", ten_god["day_ten_god_branch"])
    print("시주 천간 십성:", ten_god["hour_ten_god_stem"])
    print("시주 지지 십성:", ten_god["hour_ten_god_branch"])

    twelve_state = get_twelve_state(saju1)
    print("연주 12운성:", twelve_state["year_twelve_state"])
    print("월주 12운성:", twelve_state["month_twelve_state"])
    print("일주 12운성:", twelve_state["day_twelve_state"])
    print("시주 12운성:", twelve_state["hour_twelve_state"])

    # 음력 예시 (필요시 주석 해제)
    # saju2 = get_saju(1988, 1, 18, 15, 0, is_lunar=True, is_leap_month=False, tz=9)
    # print("\n=== 음력(1988년 1월 18일) -> 양력 변환 -> 사주 계산 ===")
    # print("연주:", "".join(saju2["year"]))
    # print("월주:", "".join(saju2["month"]))
    # print("일주:", "".join(saju2["day"]))
    # print("시주:", "".join(saju2["hour"]))
    # print("calc_info:", saju2["calc_info"])
