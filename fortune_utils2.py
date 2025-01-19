import math
import datetime
import swisseph as swe
from korean_lunar_calendar import KoreanLunarCalendar
from datetime import timedelta, timezone

# -----------------------------------------------
# 1) 간지(干支) 기초 데이터
# -----------------------------------------------
HEAVENLY_STEMS = ["갑","을","병","정","무","기","경","신","임","계"]
EARTHLY_BRANCHES = ["자","축","인","묘","진","사","오","미","신","유","술","해"]

GANZHI_60 = [
    (HEAVENLY_STEMS[i % 10], EARTHLY_BRANCHES[i % 12]) for i in range(60)
]

# 천간-천간 십성 관계 정의
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

# 일간-지지 십성 관계 정의
TEN_GODS_BRANCH_TABLE = {
    "갑": ["정인", "정재", "비견", "겁재", "편재", "식신", "상관", "정관", "편관", "편인", "정재", "편인"],
    "을": ["편인", "편재", "겁재", "비견", "정재", "상관", "식신", "편관", "정관", "정인", "편재", "정인"],
    "병": ["정관", "정재", "편인", "정인", "비견", "겁재", "식신", "상관", "편재", "편관", "정재", "편관"],
    "정": ["편관", "편재", "정인", "편인", "겁재", "비견", "상관", "식신", "정재", "정관", "편재", "정관"],
    "무": ["정재", "정재", "편관", "정관", "비견", "겁재", "식신", "상관", "편재", "정재", "편관", "정재"],
    "기": ["편재", "편재", "정관", "편관", "겁재", "비견", "상관", "식신", "정재", "편재", "정관", "편재"],
    "경": ["식신", "정재", "편재", "정재", "편인", "정인", "비견", "겁재", "식신", "상관", "정재", "상관"],
    "신": ["상관", "편재", "정재", "편재", "정인", "편인", "겁재", "비견", "상관", "식신", "편재", "식신"],
    "임": ["편재", "식신", "상관", "식신", "편관", "정관", "편인", "정인", "비견", "겁재", "식신", "겁재"],
    "계": ["식신", "상관", "편재", "상관", "정관", "편관", "정인", "편인", "겁재", "비견", "상관", "비견"]
}

# 천간과 지지의 오행 정의
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

# 12운성 정의
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

REFERENCE_DATE = datetime.datetime(1936, 2, 12,0,0,0)  # '갑자일' 기준일

def day_index_from_reference(dt: datetime.datetime) -> int:
    """REFERENCE_DATE부터 dt까지의 '일수 차이'(date 기준)."""
    return (dt.date() - REFERENCE_DATE.date()).days

# -----------------------------------------------
# 2) pyswisseph (Swiss Ephemeris) 세팅
#    - ephe 파일을 사용하려면 swe.set_ephe_path() 지정 필요
#    - 여기서는 MOSEPH(대략 계산) 사용 예시 또는 SWIEPH+ephe
# -----------------------------------------------
# swe.set_ephe_path("path/to/ephe-files")  # 필요하다면 지정
# FLAG_EXAMPLE = swe.FLG_SWIEPH | swe.FLG_SPEED
swe.set_ephe_path(r"C:\ephe")  # 에페머리스 .se1 파일 위치
FLAG_EXAMPLE = swe.FLG_SWIEPH | swe.FLG_SPEED  # ephe 없이 계산(정확도 약간↓)

def sun_ecliptic_longitude(jd: float, flags=FLAG_EXAMPLE) -> float:
    """주어진 JD에서 태양의 황경(geocentric ecliptic longitude, 0~360)"""
    (positions, ret_code) = swe.calc(jd, swe.SUN, flags)
    if ret_code < 0:
        raise ValueError(f"swe.calc error code={ret_code}, detail={positions}")
    lon, lat, dist, lon_spd, lat_spd, dist_spd = positions
    return lon

def julday_from_utc(dt_utc: datetime.datetime) -> float:
    """UTC datetime을 julday로 변환"""
    y, m, d = dt_utc.year, dt_utc.month, dt_utc.day
    h = dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600
    return swe.julday(y, m, d, h, swe.GREG_CAL)

def datetime_from_julday(jd: float, tz=0) -> datetime.datetime:
    """julday + tz(시차) -> datetime (Local)"""
    year, month, day, hour_float = swe.revjul(jd, swe.GREG_CAL)
    hh = int(hour_float)
    mm = int((hour_float - hh) * 60)
    ss = int((((hour_float - hh) * 60) - mm) * 60)
    dt_utc = datetime.datetime(year, month, day, hh, mm, ss, tzinfo=datetime.timezone.utc)
    return dt_utc + datetime.timedelta(hours=tz)

# -----------------------------------------------
# 3) 24절기 각도 목록
#    (전통적으로 입춘=315°, 우수=330°, 경칩=345°, 춘분=0°, …)
# -----------------------------------------------
SOLAR_TERMS = [
    (315.0, "입춘"), (330.0, "우수"), (345.0, "경칩"), (0.0,   "춘분"),
    (15.0,  "청명"), (30.0,  "곡우"), (45.0,  "입하"), (60.0,  "소만"),
    (75.0,  "망종"), (90.0,  "하지"), (105.0, "소서"), (120.0, "대서"),
    (135.0, "입추"), (150.0, "처서"), (165.0, "백로"), (180.0, "추분"),
    (195.0, "한로"), (210.0, "상강"), (225.0, "입동"), (240.0, "소설"),
    (255.0, "대설"), (270.0, "동지"), (285.0, "소한"), (300.0, "대한"),
]

def angle_diff(lon: float, target_deg: float) -> float:
    """
    lon - target_deg (mod 360)을 -180~+180 범위로 보정.
    """
    d = (lon - target_deg) % 360.0
    if d > 180:
        d -= 360
    return d

def find_solar_longitude_cross(target_deg: float, jd_start: float, jd_end: float, max_iter=30):
    """
    jd_start~jd_end 범위 안에서 태양 황경이 target_deg에 도달하는 시점(JD)을 이분법으로 찾는다.
    """
    v_start = angle_diff(sun_ecliptic_longitude(jd_start), target_deg)
    v_end   = angle_diff(sun_ecliptic_longitude(jd_end), target_deg)
    
    if (v_start > 0 and v_end > 0) or (v_start < 0 and v_end < 0):
        # 범위 내에서 crossing이 없으면 경고 출력 후 범위 확장
        # print(f"Warning: No crossing found in range [{jd_start}, {jd_end}] for target_deg={target_deg}. Extending range...")
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

def calc_solar_terms_of_year_swe(year: int, tz=9):
    """
    year년의 24절기 시각(로컬 tz=9)을 dict로 반환:
       {"입춘": datetime, "우수": datetime, ...}
    모식천문(MOSEPH) 또는 SWIEPH 모드.
    """
    result = {}
    # 1) year-01-01 00:00 UTC ~ (year+1)-01-01 00:00 UTC 범위
    start_utc = datetime.datetime(year,1,1,0,0,0, tzinfo=datetime.timezone.utc)
    end_utc   = datetime.datetime(year+1,1,1,0,0,0, tzinfo=datetime.timezone.utc)
    jd_start  = julday_from_utc(start_utc)
    jd_end    = julday_from_utc(end_utc)
    # print(f"JD Start: {jd_start}, JD End: {jd_end}")
    # print(f"Longitude at Start JD: {sun_ecliptic_longitude(jd_start)}")
    # print(f"Longitude at End JD: {sun_ecliptic_longitude(jd_end)}")

    for angle, name in SOLAR_TERMS:
        jd_cross = find_solar_longitude_cross(angle, jd_start, jd_end)
        dt_local = datetime_from_julday(jd_cross, tz=tz)
        
        # 결과가 같은 해에 속하는지 확인(로컬기준 year와 일치?)
        if dt_local.year == year:
            result[name] = dt_local
    
    return result

def calc_solar_terms_of_year_swe_safe(year: int, tz=9):
    try:
        terms = calc_solar_terms_of_year_swe(year, tz=tz)
    except Exception as e:
        print(f"Error in solar term calculation: {e}")
        terms = {}

    # 기본값 추가
    default_terms = {
        "입춘": datetime.datetime(year, 2, 4, 5, 0, tzinfo=timezone(timedelta(hours=tz))),
        "우수": datetime.datetime(year, 2, 19, 6, 0, tzinfo=timezone(timedelta(hours=tz))),
        # 추가 절기 기본값...
    }
    for key, value in default_terms.items():
        terms.setdefault(key, value)
    return terms

# -----------------------------------------------
# 4) 사주 계산(년주, 월주, 일주, 시주)
#    - 연주: 입춘 기준(전년도/당해년도)
#    - 월주: 절기 구간
#    - 일주: 1900-01-31 '갑자일' 기준
#    - 시주: 야자시(23시 이후 => 익일) 처리
# -----------------------------------------------
def get_year_ganzhi(year: int, after_ipchun: bool):
    """1864년 = 갑자년 기준."""
    base_year = 1864
    if not after_ipchun:
        year -= 1
    offset = (year - base_year) % 60
    return GANZHI_60[offset]

def get_month_ganzhi(year_gz_index: int, month_order: int):
    """
    month_order=1~12 (음력상 '인월=1', '묘월=2', ...)
    공식: (연간지인덱스 * 12 + (month_order-1)) % 60
    """
    offset = (year_gz_index * 12 + (month_order - 1)) % 60
    return GANZHI_60[offset]

def get_hour_ganzhi(day_gz_index: int, hour_0_23: int):
    """
    시주: (일간 * 2 + 시간대) % 10, 지지= hour//2
    """
    hour_zhi_index = (hour_0_23 // 2) % 12
    day_stem_index = day_gz_index % 10
    hour_stem_index = (day_stem_index*2 + hour_zhi_index) % 10
    return (HEAVENLY_STEMS[hour_stem_index], EARTHLY_BRANCHES[hour_zhi_index])

def find_month_order(target_dt: datetime.datetime, terms_dict: dict) -> int:
    """
    target_dt가 어느 절기 구간에 속하는지 찾아 1~12월(음력상) 반환.
    여기서는 "입춘=1월, 우수=2월, 경칩=3월..." 단순화 예시.
    """
    # 주요 12절기만 추출
    major_month_terms = [
        "입춘", "우수", "경칩", "춘분", "청명", "곡우",
        "입하", "소만", "망종", "하지", "소서", "대서"
    ]
    # (월번호=인덱스+1, 절기이름, 해당시각) 리스트
    month_list = []
    for i, tname in enumerate(major_month_terms):
        dt_ = terms_dict.get(tname, None)
        if dt_:
            month_list.append((i+1, tname, dt_))
    # 시간 순서대로 정렬
    month_list.sort(key=lambda x: x[2])

    for i in range(len(month_list)-1):
        morder, nm, dt_start = month_list[i]
        morder_next, nm_next, dt_end = month_list[i+1]
        if dt_start <= target_dt < dt_end:
            return morder
    
    # 마지막 절기보다 이후면 => 마지막 월번호(예: 12)
    if month_list:
        return month_list[-1][0]
    return 1  # fallback

def calculate_day_pillar(dt_local: datetime.datetime):
    """
    기준일로부터 일수를 계산하여 일주(干支)를 반환.
    """
    
    # 야자시 처리 (23시 이후 익일로 변경)
    if dt_local.hour >= 23:
        dt_local += timedelta(days=1)
        dt_local = dt_local.replace(hour=dt_local.hour - 24)
    
    # 기준일로부터 일수 차이
    day_diff = (dt_local.date() - REFERENCE_DATE.date()).days
    day_gz_index = day_diff % 60
    day_gan, day_zhi = GANZHI_60[day_gz_index]
    
    # print(f"기준일로부터 일수 차이: {day_diff}")
    # print(f"일주 간지 인덱스: {day_gz_index}, 간지: {day_gan}{day_zhi}")
    
    return day_gan, day_zhi

def calculate_twelve_state(day_stem, branch):
    """
    주어진 일간과 지지에 따른 12운성을 계산합니다.

    Parameters:
        day_stem (str): 일간 (천간, 예: "갑")
        branch (str): 지지 (예: "자", "축")

    Returns:
        str: 12운성 (예: "장생", "관대", ...)
    """
    # 12운성 테이블에서 일간에 해당하는 배열 가져오기
    if day_stem not in TWELVE_STATES_TABLE:
        raise ValueError(f"일간 '{day_stem}'이 올바르지 않습니다.")
    states = TWELVE_STATES_TABLE[day_stem]

    # 지지가 12운성 순서에 따라 몇 번째에 해당하는지 확인
    if branch not in EARTHLY_BRANCHES:
        raise ValueError(f"지지 '{branch}'이 올바르지 않습니다.")
    branch_index = EARTHLY_BRANCHES.index(branch)

    # 해당 지지의 12운성 반환
    return states[branch_index]

def get_saju(year, month, day, hour=0, minute=0, is_lunar=False, is_leap_month=False, tz=9):
    # (A) 음력 => 양력 변환 (korean_lunar_calendar)
    if is_lunar:
        cal = KoreanLunarCalendar()
        cal.setLunarDate(year, month, day, is_leap_month)
        solar_y = cal.getSolarYear()
        solar_m = cal.getSolarMonth()
        solar_d = cal.getSolarDay()
    else:
        solar_y, solar_m, solar_d = year, month, day

    # (B) datetime(로컬) 만들기
    kst = timezone(timedelta(hours=tz))  # KST (UTC+9)
    dt_local = datetime.datetime(solar_y, solar_m, solar_d, hour, minute, tzinfo=kst)

    # (C) 24절기 계산 (그 해)
    year_terms = calc_solar_terms_of_year_swe_safe(dt_local.year, tz=tz)

    # (D) 입춘
    ipchun_dt = year_terms.get("입춘", None)
    if not ipchun_dt:
        ipchun_dt = datetime.datetime(dt_local.year, 2, 4, 5, 0, tzinfo=kst)  # 대략적인 값

    # (E) 연주(Year Pillar) 결정
    after_ipchun = (dt_local >= ipchun_dt)  # offset-aware끼리 비교 가능
    year_gan, year_zhi = get_year_ganzhi(dt_local.year, after_ipchun)
    # 연간지 index
    year_gz_index = ((dt_local.year - 1864) - (0 if after_ipchun else 1)) % 60
    # 연주 오행
    year_gan_element = ELEMENTS.get(year_gan, "")
    year_zhi_element = ELEMENTS.get(year_zhi, "")

    # 월주
    m_order = find_month_order(dt_local, year_terms)
    month_gan, month_zhi = get_month_ganzhi(year_gz_index, m_order)
    # 월주 오행
    month_gan_element = ELEMENTS.get(month_gan, "")
    month_zhi_element = ELEMENTS.get(month_zhi, "")

    # 일주
    day_diff = day_index_from_reference(dt_local)
    day_gz_index = day_diff % 60
    day_gan, day_zhi = calculate_day_pillar(dt_local)
    # 일주 오행
    day_gan_element = ELEMENTS.get(day_gan, "")
    day_zhi_element = ELEMENTS.get(day_zhi, "")

    # 시주
    hour_gan, hour_zhi = get_hour_ganzhi(day_gz_index, dt_local.hour)
    hour_gan_element = ELEMENTS.get(hour_gan, "")
    hour_zhi_element = ELEMENTS.get(hour_zhi, "")

    return {
        "year": (year_gan, year_zhi),
        "year_element": (year_gan_element, year_zhi_element),
        "month": (month_gan, month_zhi),
        "month_element": (month_gan_element, month_zhi_element),
        "day": (day_gan, day_zhi),
        "day_element": (day_gan_element, day_zhi_element),
        "hour": (hour_gan, hour_zhi),
        "hour_element": (hour_gan_element, hour_zhi_element),
        "calc_info": {
            "final_solar_date": dt_local.strftime("%Y-%m-%d %H:%M"),
            "ipchun_date": ipchun_dt.strftime("%Y-%m-%d %H:%M") if isinstance(ipchun_dt, datetime.datetime) else str(ipchun_dt),
            "after_ipchun": after_ipchun,
            "month_order": m_order,
            "is_lunar_input": is_lunar,
            "is_leap_month": is_leap_month,
        }
    }

def get_ten_god_stem(day_stem, other_stem):
    """
    주어진 일간과 다른 천간 간의 십성을 반환합니다.

    Parameters:
        day_stem (str): 일간의 천간
        other_stem (str): 비교할 다른 천간

    Returns:
        str: 십성 (예: "비견", "편재" 등)
    """
    return TEN_GODS_STEM_TABLE[day_stem][HEAVENLY_STEMS.index(other_stem)]

def get_ten_god_branch(day_stem, branch):
    """
    주어진 일간과 지지 간의 십성을 반환합니다.

    Parameters:
        day_stem (str): 일간의 천간
        branch (str): 비교할 지지

    Returns:
        str: 십성 (예: "정재", "편관" 등)
    """
    return TEN_GODS_BRANCH_TABLE[day_stem][EARTHLY_BRANCHES.index(branch)]

def get_ten_god(saju):
    day_stem = saju["day"][0]
    year_ten_god_stem = get_ten_god_stem(day_stem, saju["year"][0])
    year_ten_god_branch = get_ten_god_branch(day_stem, saju["year"][1])
    month_ten_god_stem = get_ten_god_stem(day_stem, saju["month"][0])
    month_ten_god_branch = get_ten_god_branch(day_stem, saju["month"][1])
    day_ten_god_stem = "일원"
    day_ten_god_branch = get_ten_god_branch(day_stem, saju["day"][1])
    hour_ten_god_stem = get_ten_god_stem(day_stem, saju["hour"][0])
    hour_ten_god_branch = get_ten_god_branch(day_stem, saju["hour"][1])

    return {
        "year_ten_god_stem": year_ten_god_stem,
        "year_ten_god_branch": year_ten_god_branch,
        "month_ten_god_stem": month_ten_god_stem,
        "month_ten_god_branch": month_ten_god_branch,
        "day_ten_god_stem": day_ten_god_stem,
        "day_ten_god_branch": day_ten_god_branch,
        "hour_ten_god_stem": hour_ten_god_stem,
        "hour_ten_god_branch": hour_ten_god_branch
    }

def calculate_twelve_state(day_stem, branch):
    """
    주어진 일간과 지지에 따른 12운성을 계산합니다.

    Parameters:
        day_stem (str): 일간 (천간, 예: "갑")
        branch (str): 지지 (예: "자", "축")

    Returns:
        str: 12운성 (예: "장생", "관대", ...)
    """
    # 12운성 테이블에서 일간에 해당하는 순서 가져오기
    if day_stem not in TWELVE_STATES_TABLE:
        raise ValueError(f"일간 '{day_stem}'이 올바르지 않습니다.")
    branches = TWELVE_STATES_TABLE[day_stem]

    # 지지가 12운성 순서에 따라 몇 번째에 해당하는지 확인
    if branch not in EARTHLY_BRANCHES:
        raise ValueError(f"지지 '{branch}'이 올바르지 않습니다.")
    branch_index = EARTHLY_BRANCHES.index(branch)

    # 해당 지지의 12운성 반환
    return branches[branch_index]

def get_twelve_state(saju):
    day_stem = saju["day"][0]
    year_twelve_state = calculate_twelve_state(day_stem, saju["year"][1])
    month_twelve_state = calculate_twelve_state(day_stem, saju["month"][1])
    day_twelve_state = calculate_twelve_state(day_stem, saju["day"][1])
    hour_twelve_state = calculate_twelve_state(day_stem, saju["hour"][1])

    return {
        "year_twelve_state": year_twelve_state,
        "month_twelve_state": month_twelve_state,
        "day_twelve_state": day_twelve_state,
        "hour_twelve_state": hour_twelve_state
    }


# -----------------------------------------------
# 5) 테스트 예시
# -----------------------------------------------
if __name__ == "__main__":

    # A. 양력 예시: 1988-02-04 23:30
    #    (실제 입춘 무렵)
    saju1 = get_saju(1997,3,24,17,00, is_lunar=False, tz=9)
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
    print("일간:", "".join(saju1["day"][0]))
    print("연주 천간 십성:", "".join(ten_god["year_ten_god_stem"]))
    print("연주 지지 십성:", "".join(ten_god["year_ten_god_branch"]))
    print("월주 천간 십성:", "".join(ten_god["month_ten_god_stem"]))
    print("월주 지지 십성:", "".join(ten_god["month_ten_god_branch"]))
    print("일주 천간 십성:", "".join(ten_god["day_ten_god_stem"]))
    print("일주 지지 십성:", "".join(ten_god["day_ten_god_branch"]))
    print("시주 천간 십성:", "".join(ten_god["hour_ten_god_stem"]))
    print("시주 지지 십성:", "".join(ten_god["hour_ten_god_branch"]))
    twelve_state = get_twelve_state(saju1)
    print("연주 12운성:", "".join(twelve_state["year_twelve_state"]))
    print("월주 12운성:", "".join(twelve_state["month_twelve_state"]))
    print("일주 12운성:", "".join(twelve_state["day_twelve_state"]))
    print("시주 12운성:", "".join(twelve_state["hour_twelve_state"]))

    # terms_dict = calc_solar_terms_of_year_swe_safe(1997, tz=9)
    # print("절기 시각:", terms_dict)


    # B. 음력 예시: (가상의) 1988년 1월 18일, 윤달 여부= False
    #    실제 윤달 예: 1999년 윤5월 등...
    # saju2 = get_saju(1988,1,18,15,0, is_lunar=True, is_leap_month=False, tz=9)
    # print("\n=== 음력(1988년 1월 18일) => 양력 변환 => 사주 계산 ===")
    # print("연주:", "".join(saju2["year"]))
    # print("월주:", "".join(saju2["month"]))
    # print("일주:", "".join(saju2["day"]))
    # print("시주:", "".join(saju2["hour"]))
    # print("calc_info:", saju2["calc_info"])
