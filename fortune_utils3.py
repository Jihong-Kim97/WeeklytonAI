import math
import datetime
from datetime import timedelta, timezone
from korean_lunar_calendar import KoreanLunarCalendar
from pymeeus.Epoch import Epoch
from pymeeus.Sun import Sun

# -----------------------------------------------
# 1) 간지(干支) 기초 데이터
# -----------------------------------------------
HEAVENLY_STEMS = ["갑", "을", "병", "정", "무", "기", "경", "신", "임", "계"]
EARTHLY_BRANCHES = ["자", "축", "인", "묘", "진", "사", "오", "미", "신", "유", "술", "해"]

GANZHI_60 = [
    (HEAVENLY_STEMS[i % 10], EARTHLY_BRANCHES[i % 12]) for i in range(60)
]

# 천간-천간, 일간-지지 십성 테이블 (생략없이 동일하게 사용)
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
# 2) 천문 계산 (PyMeeus 사용)
# -----------------------------------------------
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
    JD = (math.floor(365.25 * (year + 4716))
          + math.floor(30.6001 * (month + 1))
          + day + B - 1524.5 + hour / 24.0)
    return JD

def datetime_from_julday(jd: float, tz=0) -> datetime.datetime:
    """Julian Day와 시차(tz)를 이용하여 로컬 datetime 반환"""
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
    """PyMeeus를 사용하여 주어진 JD에서 태양의 황경(0~360°) 계산"""
    dt_utc = datetime_from_julday(jd, tz=0)
    hour_decimal = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
    e = Epoch(dt_utc.year, dt_utc.month, dt_utc.day, hour_decimal)
    s = Sun(e)
    return s.apparent_longitude % 360.0

def angle_diff(lon: float, target_deg: float) -> float:
    """lon과 target_deg의 차이를 -180 ~ +180 범위로 반환"""
    d = (lon - target_deg) % 360.0
    if d > 180:
        d -= 360
    return d

def find_solar_longitude_cross(target_deg: float, jd_start: float, jd_end: float, max_iter=30):
    """jd_start ~ jd_end 범위에서 태양 황경이 target_deg에 도달하는 JD를 이분법으로 찾음"""
    v_start = angle_diff(sun_ecliptic_longitude(jd_start), target_deg)
    v_end = angle_diff(sun_ecliptic_longitude(jd_end), target_deg)
    
    if (v_start > 0 and v_end > 0) or (v_start < 0 and v_end < 0):
        return find_solar_longitude_cross(target_deg, jd_start - 30, jd_end + 30, max_iter)
    
    for _ in range(max_iter):
        jd_mid = (jd_start + jd_end) / 2.0
        v_mid = angle_diff(sun_ecliptic_longitude(jd_mid), target_deg)
        if abs(v_mid) < 1e-8:
            return jd_mid
        if (v_start > 0 and v_mid > 0) or (v_start < 0 and v_mid < 0):
            jd_start = jd_mid
            v_start = v_mid
        else:
            jd_end = jd_mid
            v_end = v_mid
    return (jd_start + jd_end) / 2.0

def calc_solar_terms_of_year(year: int, tz=9):
    """
    주어진 연도의 24절기 시각(로컬 tz 기준)을 dict로 반환.
    예: {"입춘": datetime, "우수": datetime, ...}
    """
    # 1월 1일 ~ 다음 해 1월 1일 사이의 절기 시각 계산
    start_utc = datetime.datetime(year, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    end_utc = datetime.datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    jd_start = julday_from_utc(start_utc)
    jd_end = julday_from_utc(end_utc)

    # 24절기 (여기서는 주요 12절기만 사용)
    SOLAR_TERMS = [
        (315.0, "입춘"), (330.0, "우수"), (345.0, "경칩"), (0.0, "춘분"),
        (15.0, "청명"), (30.0, "곡우"), (45.0, "입하"), (60.0, "소만"),
        (75.0, "망종"), (90.0, "하지"), (105.0, "소서"), (120.0, "대서")
    ]
    result = {}
    for angle, name in SOLAR_TERMS:
        jd_cross = find_solar_longitude_cross(angle, jd_start, jd_end)
        dt_local = datetime_from_julday(jd_cross, tz=tz)
        if dt_local.year == year:
            result[name] = dt_local
    return result

def calc_solar_terms_of_year_safe(year: int, tz=9):
    try:
        terms = calc_solar_terms_of_year(year, tz=tz)
    except Exception as e:
        print(f"Error in solar term calculation: {e}")
        terms = {}
    default_terms = {
        "입춘": datetime.datetime(year, 2, 4, 5, 0, tzinfo=timezone(timedelta(hours=tz))),
        "우수": datetime.datetime(year, 2, 19, 6, 0, tzinfo=timezone(timedelta(hours=tz))),
    }
    for key, value in default_terms.items():
        terms.setdefault(key, value)
    return terms

# -----------------------------------------------
# 3) 월주 계산 (정밀 보정 적용)
#
# 1. 우선, 생시가 속한 절기 구간(주요 12절기)와 그 구간 내 진행 비율(fraction)을 구한다.
# 2. 기본 월 순서는 해당 절기의 순서로 결정되며, 만약 진행 비율이 0.5 이상이면
#    실제 월주 계산 시 '다음 월'로 보정한다.
#
# 월간은: (연간 천간 인덱스 × 2 + 보정된 월 순번) mod 10  
# 월지(월의 지지)는 고정 순서: [인, 묘, 진, 사, 오, 미, 신, 유, 술, 해, 자, 축]
# -----------------------------------------------
def get_month_order_and_fraction(target_dt: datetime.datetime, terms_dict: dict):
    """
    target_dt가 속한 주요 절기 구간(월 구간)과, 해당 구간 내 진행 비율(0~1)을 반환.
    절기 목록은 ["입춘", "우수", "경칩", "춘분", "청명", "곡우", "입하", "소만", "망종", "하지", "소서", "대서"]를 사용.
    """
    major_month_terms = [
        "입춘", "우수", "경칩", "춘분", "청명", "곡우",
        "입하", "소만", "망종", "하지", "소서", "대서"
    ]
    month_list = []
    for i, tname in enumerate(major_month_terms):
        dt_term = terms_dict.get(tname)
        if dt_term:
            month_list.append((i+1, tname, dt_term))
    month_list.sort(key=lambda x: x[2])
    
    # target_dt가 어느 구간에 속하는지
    for i in range(len(month_list)-1):
        morder, tname, start_dt = month_list[i]
        _, _, end_dt = month_list[i+1]
        if start_dt <= target_dt < end_dt:
            fraction = (target_dt - start_dt).total_seconds() / (end_dt - start_dt).total_seconds()
            return morder, fraction
    if month_list:
        # 만약 마지막 절기 이후라면
        morder, tname, start_dt = month_list[-1]
        return morder, 0.0
    return 1, 0.0

def get_month_ganzhi_correct(year_gan: str, month_order: int):
    """
    보정된 월 순번(month_order)을 이용하여 월주(월간, 월지)를 계산.
    월간 = (연간 천간 인덱스 × 2 + 보정된 월 순번) mod 10  
    월지 = 순서: [인, 묘, 진, 사, 오, 미, 신, 유, 술, 해, 자, 축]
    """
    y_index = HEAVENLY_STEMS.index(year_gan)
    month_stem_index = (y_index * 2 + month_order) % 10
    month_branch_order = ["인", "묘", "진", "사", "오", "미", "신", "유", "술", "해", "자", "축"]
    month_branch = month_branch_order[(month_order - 1) % 12]
    return (HEAVENLY_STEMS[month_stem_index], month_branch)

# -----------------------------------------------
# 4) 사주 계산 (년주, 월주, 일주, 시주)
#    - 연주는 입춘 기준 (입춘 이후이면 당해, 그 전이면 전년도)
#    - 월주는 위에서 보정한 절기 진행 비율을 반영하여 결정
#    - 일주는 기준일(1936-02-12, 갑자일) 기준
#    - 시주는 23시 이후는 익일로 처리
# -----------------------------------------------
def get_year_ganzhi(year: int, after_ipchun: bool):
    """1864년을 갑자년 기준으로 연주 계산"""
    base_year = 1864
    if not after_ipchun:
        year -= 1
    offset = (year - base_year) % 60
    return GANZHI_60[offset]

def get_hour_ganzhi(day_gz_index: int, hour_0_23: int):
    """
    시주는 (일간 인덱스 × 2 + 시간대) mod 10, 
    여기서 시간대는 2시간 단위 (예: 0~1시→0, 2~3시→1, …)
    """
    hour_zhi_index = (hour_0_23 // 2) % 12
    day_stem_index = day_gz_index % 10
    hour_stem_index = (day_stem_index * 2 + hour_zhi_index) % 10
    return (HEAVENLY_STEMS[hour_stem_index], EARTHLY_BRANCHES[hour_zhi_index])

def calculate_day_pillar(dt_local: datetime.datetime):
    """
    기준일로부터 일수를 계산하여 일주(일간, 일지)를 반환.
    단, 23시 이후는 익일로 처리.
    """
    if dt_local.hour >= 23:
        dt_local += timedelta(days=1)
        dt_local = dt_local.replace(hour=dt_local.hour - 24)
    day_diff = (dt_local.date() - REFERENCE_DATE.date()).days
    return GANZHI_60[day_diff % 60]

def calculate_twelve_state(day_stem, branch):
    """
    주어진 일간과 지지에 따른 12운성을 반환.
    """
    if day_stem not in TWELVE_STATES_TABLE:
        raise ValueError(f"일간 '{day_stem}'이 올바르지 않습니다.")
    states = TWELVE_STATES_TABLE[day_stem]
    if branch not in EARTHLY_BRANCHES:
        raise ValueError(f"지지 '{branch}'이 올바르지 않습니다.")
    branch_index = EARTHLY_BRANCHES.index(branch)
    return states[branch_index]

def get_saju(year, month, day, hour=0, minute=0, is_lunar=False, is_leap_month=False, tz=9):
    # (A) 음력 입력이면 양력으로 변환 (korean_lunar_calendar 사용)
    if is_lunar:
        cal = KoreanLunarCalendar()
        cal.setLunarDate(year, month, day, is_leap_month)
        solar_y = cal.getSolarYear()
        solar_m = cal.getSolarMonth()
        solar_d = cal.getSolarDay()
    else:
        solar_y, solar_m, solar_d = year, month, day

    # (B) 로컬 datetime 생성
    kst = timezone(timedelta(hours=tz))
    dt_local = datetime.datetime(solar_y, solar_m, solar_d, hour, minute, tzinfo=kst)

    # (C) 해당 연도의 24절기 계산
    year_terms = calc_solar_terms_of_year_safe(dt_local.year, tz=tz)

    # (D) 입춘 기준 (절기 정보 없으면 기본값 사용)
    ipchun_dt = year_terms.get("입춘", None)
    if not ipchun_dt:
        ipchun_dt = datetime.datetime(dt_local.year, 2, 4, 5, 0, tzinfo=kst)

    # (E) 연주 결정: 입춘 이후이면 당해, 아니면 전년도
    after_ipchun = (dt_local >= ipchun_dt)
    year_gan, year_zhi = get_year_ganzhi(dt_local.year, after_ipchun)

    # (F) 월주 결정: 절기 구간 내 진행률을 반영하여 보정
    base_m_order, fraction = get_month_order_and_fraction(dt_local, year_terms)
    # 만약 진행률이 0.5 이상이면 '다음 월'로 보정
    if fraction >= 0.5:
        m_order_adj = (base_m_order % 12) + 1
    else:
        m_order_adj = base_m_order
    month_gan, month_zhi = get_month_ganzhi_correct(year_gan, m_order_adj)

    # (G) 일주 결정
    day_gan, day_zhi = calculate_day_pillar(dt_local)

    # (H) 시주 결정
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
            "base_month_order": base_m_order,
            "fraction_in_month": round(fraction, 3),
            "adjusted_month_order": m_order_adj,
            "is_lunar_input": is_lunar,
            "is_leap_month": is_leap_month,
        }
    }

def get_ten_god_stem(day_stem, other_stem):
    """일간과 다른 천간 간의 십성 반환"""
    return TEN_GODS_STEM_TABLE[day_stem][HEAVENLY_STEMS.index(other_stem)]

def get_ten_god_branch(day_stem, branch):
    """일간과 지지 간의 십성 반환"""
    return TEN_GODS_BRANCH_TABLE[day_stem][EARTHLY_BRANCHES.index(branch)]

def get_ten_god(saju):
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

def get_twelve_state(saju):
    day_stem = saju["day"][0]
    return {
        "year_twelve_state": calculate_twelve_state(day_stem, saju["year"][1]),
        "month_twelve_state": calculate_twelve_state(day_stem, saju["month"][1]),
        "day_twelve_state": calculate_twelve_state(day_stem, saju["day"][1]),
        "hour_twelve_state": calculate_twelve_state(day_stem, saju["hour"][1])
    }

# -----------------------------------------------
# 5) 테스트 예시
# -----------------------------------------------
if __name__ == "__main__":
    # 예시: 1997년 3월 24일 17:00 KST (입춘 무렵)
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
