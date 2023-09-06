# ����OPM

�������У� flow NORNE_ATW2013.DATA --output-dir=out_serial

�������У� mpirun --allow-run-as-root -np 8 NORNE_ATW2013.DATA --output-dir=out_parallel

������Ϻ󣬽����ɼ����µ��ļ���

NORNE.EGRID
NORNE.INIT
NORNE.SMSPEC
NORNE.UNRST
NORNE.UNSMRY

## ���н��

���ɼ�����־�ļ����ն������PRT�ļ���DBG�ļ�

�ն���������������ⲻ������Լ�����ƻ������������С����ʱ�䲽����

Problem: Solver convergence failure - Iteration limit reached
Timestep chopped to 3.630000 days

## PRT log(NORNE_ATW2013.PRT)

���ն������ϸ�ڵ���Ϣ

## DBG log (NORNE_ATW2013.DBG)

��PRT�����ϸ�ڵ���Ϣ������debuging��Ϣ

# ���ƾ�����

## ������

ʹ��[libecl����](https://github.com/Statoil/libecl)�е�ecl_summary����

ecl_summary --list NORNE_ATW2013.DATA

Keywords start with a letter signifying its scope: C for completion (well perforation), F for field, G for group (of wells), R for region and W for wells.

The rest of the keyword identifies the quantity, some examples (note that some of the concepts such as oil production rate makes sense and can be used with several of the scopes):

Well keywords (W): BHP for bottom hole pressures, GOR for gas-oil ratio, OPR for oil production rate, WIR for water injection rate.

Region keywords (R): OIP for oil in place, GIPL for gas in place in the liquid phase (dissolved in it).

The string after the colon (if any) is the name of the well or group, the region number, or (for completion data) the well name and completion location.

��õ���ͬ�������б����бȽ���Ȥ���У�

WBHP:INJ �C Bottom hole pressure of INJ well
WGIR:INJ �C Gas injection rate of INJ well
WBHP:PROD �C Bottom hole pressure of PROD well
WOPR:PROD �C Oil production rate of PROD well
WGPR:PROD �C Gas production rate of PROD well

�ֶ�������ֵ�� 

ecl_summary SPE1CASE1.DATA WBHP:INJ

ecl_summary NORNE_ATW2013 WBHP:C-4H   # �鿴��C-4H�ĵײ���ѹ��

## ���Ʊ���

ʹ��[opm-utilities](https://github.com/OPM/opm-utilities/)��summaryplot��Python�ű�����

wget https://raw.githubusercontent.com/OPM/opm-utilities/master/summaryplot

apt-get install python-ecl python-numpy python-matplotlib libecl-dev

���ƾ����ߣ� 

python summaryplot WBHP:INJ WBHP:PROD WOPR:PROD WGPR:PROD WGIR:INJ SPE1CASE1.DATA

����![ͼƬ](./media/wbhp_inj.png)��ע�⣺��Ϊ�����deck����FIELD��λ(ѹ����λ��psi)����Flow���Ҳʹ�øõ�λ
