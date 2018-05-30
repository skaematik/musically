import React, { Component } from 'react';
import logo from './bars.svg';
import './dropzone.css';
import './App.css';
import upload_icon from './upload.svg';
import play_icon from './play.svg';
import pause_icon from './pause.svg';
import axios from 'axios';
import settings from './settings.json';

class App extends Component {
  constructor(props) {
    super(props);
    this.state = this.initialState();
    this.onSelectedSheetChange = this.onSelectedSheetChange.bind(this);
    this.renderOsmd = this.renderOsmd.bind(this);
    this.resetOsmd = this.resetOsmd.bind(this);
    this.renderUploader = this.renderUploader.bind(this);
    this.toggleUploader = this.toggleUploader.bind(this);
    this.togglePlay = this.togglePlay.bind(this);
    this.midiPrefix = "data:audio/midi;base64,";
    // default midi
    this.midi = `TVRoZAAAAAYAAQAHAYBNVHJrAAAAugD/AxhPbGQgTWFjRG9uYWxkIEhhZCBBIEZhcm0A/wIxQ29
      weXJpZ2h0IKkgMTk5NiBieSBEaXZlcnNpZmllZCBTb2Z0d2FyZSBSZXNlYXJjaAD/ARRCcmVudC
      BCYWlsZXk6IFBpYW5vCgD/ARlCaWxsIEJhc2hhbTogQmFzcywgRHJ1bXMKAP9UBWAAAwAAAP8hA
      QAA8AV+fwkB9wD/IQEAAP9YBAQCGAgA/1kCAAAA/1EDCSfAAP8vAE1UcmsAABUqAP8hAQAA/wMK
      UGlhbm8gTGVmdACwXRUA4C4/gwLAAAGwB38ACgCJApBKQQBHSgNDUQE+SQI7NQM3PoFAsEB/XZA
      +AAdDAA1HAAtKAAE3AC07AIFjSkcAT0YDQ0cDO0YAPjU9QwAKTwACSgAEOwAGPgBtTFMAT04DQ1
      MEsEAAAJA8TgJAPIE6sEB/QJBPAAZAAAFDABJMAA08AGdKSQFPTAJDSQM+Owc7SAawQACBJpBDA
      AGwQH8MkEoABjsADU8ABz4AgUawQAANkDY4ATxEA1FRAEpTAT41AUVJMz4ABDwAA0UAATYAA0oA
      CVEAGTw8ATYuA0pLAEVJAT4vAFFFLkoAAkUACTYAAFEABjwAAD4AgRNRTQBKTwFFTwI8SAE+OwM
      2PixFAABKAAM2AAE+AAA8AANRABlFOQBRMwI8NQBKRwE2KgE+KyxFAARKAAZRAAM2AAY8AAE+AI
      EJSk8AUUUBRU8GPjIAPDwBNjglRQAGSgAHUQACNgAEPgAAPACBEFFLAUVRAD49AUpRATxGADY+g
      S1KAABFAAlRAAQ8AAA2AAM+AIRsOywENykDPikBRzgASjEBQzWBdbBAf4E2kDcAIkoAB0MADUcA
      ATsACz4AZDs0BD4yADctAUpHAUNJAUdOUzsAAkcABEMAAzcABj4ABEoAXLBAAAqQPDIATEcBQ0M
      ASEQCQDQBNyyBVrBAfzmQTAADNwAGQwAaQAATPAA+SAAQSjMDQzUARz4EOysBPiYBNyUJsEAAgS
      pAfwmQSgACRwAANwACQwAJOwAHPgCBULBAAACQR0ADQ0cBPj8DOykCMimBUbBAfxmQMgAUPgATQ
      wANOwAiRwBESEYAPCoDRUsDPjsHMikPsEAAgVNAfySQMgAcSAACPgAFPAAgRQBKQ0sAR0ADsEAA
      A5A+NwM7KgQ3MIE8sEB/N0AAgRdAf2WQPgAAQwARRwAZNwAgOwBzsEAAYJBKPQZHSARDPQI+KgQ
      7LQQ3KoMJSgAGsEB/BJBHABZDABA3ACw7AAQ+AG1PRABKRQJDQwE+NAA7PmNDAA1PAAZKAAA+AA
      M7AECwQAAKkE82BExHAjw2AENJAzcvBEAygVmwQH82kDcADUMABE8ACUAADEwAGjwALUpDAU9GB
      UNBBDshAzcrAT4qCbBAAIEpQH8KkE8ABzsAAkoABDcAAj4ABEMAgUewQAAJkEdCBEo3A0NHADcu
      Az45ADsrAjIggUewQH8GkDcABzIADEcAC0MAADsAJj4ANkoALUhGBEVJArBAAAGQPjMEOTgFPC0
      JMiWBUbBAfxOQRQABSAADMgAJPgAPPAAaOQBksEAAD5A3LgRDTwBHRgI+RR47GIMCsEB/gSeQPg
      AGQwADsEAABJBHAAM3AAY7AIFaR1gBPk8CN0gAOzgBQ1t5QwAAPgAGRwAGNwAKOwCFFEdaAkNfA
      D5VAztKBjdQfj4AAEMACUcABDcACTsAhRdHVgA7RAFKUQFDWwI+UQI3TG9DAAI+AAU3AARHAABK
      AAE7AIF7O0IBR1YDPlEAQ10BN06BCzcAA0MAAT4AAzsAAkcAgg47RANHVgA+TQA3SgFDW3tDAAQ
      +AANHAAI3AAo7AIF0Q1MCR0wBN0gAO0ACPkc9PgABNwABQwACOwAGRwB9R0IGQ00BPkECNzoBOy
      8vQwAEPgAGRwADOwABNwCBBkdOA0NVAT5HAUpJATs+ATdGX7BAf3OQPgAHQwAKRwATSgAANwAcO
      wCCYDs6AT41A0pPADczAU9SAENPQUMACzsABjcACU8ABEoAAj4AZE9QAzxCAUxVAENTBEA4ADdA
      CLBAAIEzQH9EkEMAAjcACkwAAE8AGkAAFjwAVE9MAkpLADszATcvAD4wBENLBbBAAIEmQH8RkE8
      AAkoABD4AADsABzcAAEMAgU9KNwFHSgM3PgE+PwA7Pg6wQACBN5A3AAmwQH8ekDsALEoAAz4AGk
      cAUEhGBD49AEVJAzwxATYdATkzBDIjD7BAAIFAkD4AAUgAAzIABzYAALBAfwKQRQAHPAADOQCBL
      LBAAAKQR0ICQ0cEPjcCNyMDOyyEJj4ABEMADUcADTcAGTsAgUNPPAY7MwFDRwBKRQI+MQE3LYFG
      sEB/VpBDAA1KABNPABo7AB03AAM+AIFdVkIDT0gBSj0DQzUDOzQGNzA5QwAEVgABTwAANwACOwA
      BSgBpsEAADZA8LwFUTgFITAFPUgBALgM3LYFWsEB/KpA3AANIAAxPACdAABM8AERUAAZTOwNKMA
      FPQAA+LQM3KwA7MA+wQACBMZBTAAJPAAewQH8KkEoAGTsAAzcAFz4AgRNPRgdKRwFHRAA+LgU7N
      AOwQACBV0B/D5BHAAQ7AABKAARPAA8+AIENTi4CSjMDNgwBPCEDRSoGsEAAB5A+HIE8SgAEPAAG
      TgAANgADRQAGsEB/CJA+AIECsEAAKZBMRwQ8MABIRAY3KgFAJSxIAAk3AARAAAA8AAFMAHtMNwR
      IOgA8MwZALAE3KzU3AAdAAAFIAAVMABM8AGpKKgNHLQQ7KgY+KQM3J4EwsEB/HZBHAANKABo3AA
      Y7ABY+AIEIsEAAAJBPNAJKOQI7MAE+LwFDMgA3LYI8sEB/bpA3AChPAARKAAZDAB0+AAw7AFdPS
      gJKSwI7RABDSwE+MmRDAARKAANPAA07AAE+AFw8KgBMTQNPSAFDSQBALQE3KRuwQACBD0B/XpBD
      AA1PAAY3AANMAAxAAA08AE1KLghPKQA+MAI3LABDPQM7LySwQABPQH8tkEMAEEoAEDcABE8AAjs
      ACj4AgVRTSQBPSgBKPwE7NAI3OBCwQACBDUB/VZBPAAM3AAY7AAJTAAlKAIEDTiMGUScBSCwCPC
      kDNx4dsEAAgQNAfxOQNwANUQABTgAMSAAKPACBDbBAACaQPCkASEABQCMCTzoBRS4DNx8mNwADR
      QABSAACPAADTwAHQACBB0U1AE8vAUhGBDw0AUAsBjcrLEgAAkUACUAABjcACTwAcUcyAz4sAEM5
      AzssAjcnerBAf1eQOwADNwATPgAGRwAXTwAJQwBTsEAAJJBKSQRHUgM7QgA+NAFDQwE3NnpDAAR
      HAAZKAAM7AAQ3AAY+AIUqSk0DR1IAQ0kBPjAAOzUBNzRtQwAERwAGSgAAOwADNwABPgCFL0pNAz
      4zATtCAENNAjc4AEdUcEcAA0MABDsAADcAA0oABz4Aggw7QgE+NQJKSwE3NgFHUgFDT3NDAAFHA
      AM3AANKAAE7AAk+AIIQSkkAR1ICQ00FOzwCPjEHNzhzRwABSgAAQwAFNwADOwAEPgCFIEpNAkdU
      AjtAAENPAz4yADc8aEcAAUMABTsAAjcAAj4AAEoAhUBPUgI+OQBKTwA7RgFDUwQ3QGZKAANDAAF
      PAAI3AAA7AAk+AIU6T1IGPi8BSlMBOzQAQ1cBNzxAQwAROwACNwABSgABTwAFPgCCHU9QAUpNAk
      NRAztEAzc2BD4tUzcAAEMABkoAAD4AAjsAAk8AgilPTgJKTwBDUQU7PgM3NQA+NF9KAAFDAAY3A
      AFPAAI7AAM+AIU9Sj0DTzoBQz0APjECOzEBNy2BHLBAf32QQwAKSgATTwAZOwANNwAGPgCBflNN
      BEo5AUMyAT4vADswBDcqODcAAj4ACUMAADsACkoAdlRQAkw/BDw1AEAzAzcvAbBAAAmQUwCBFrB
      AfzOQNwAgQAANTAAJPACBA1M/A0onAT4pBzssAzcnA7BAAAWQVABUsEB/XZBTAAlKAAY7AAo3AA
      M+AIFET0YBRzgFPisDMiUGsEAAgUpAfyCQMgACRwALTwAHPgB9TicPRScCPB0LMh4AsEAAgSOQM
      gATsEB/BJBOAAlFAAQ8AIEMsEAAMZBMQwNPOAFIRgE8LARAKAY3Jy1IAAlMAAo3AAM8AAhAAGVM
      PQE8MwBIRgZAKwM3Ky1IAAJMAARAAAA3AAk8AH5KKQZHMgM7LAE+KQU3KYEksEB/IJBHAABKAA8
      3AAFPAAM+AAE7AIE/sEAAAZBWNgJPQgFKMwI+KgA7KQRDMQU3KoFXsEB/gT2QNwABOwAJPgAKQw
      AJTwAESgAcVgB2U0UASjUBQzEAVkQBPjECOzYAT0YANzAsSgAAUwAHPgAATwAGNwADQwAEOwAAV
      gCBEDw6AVhFAUxDAVRIALBAAAGQQDMANzSBJLBAfy2QVAABWAAFNwAXTAAJQAA+PABGUzkDSiwB
      VicCPi4BNysCOzABsEAAY0B/SZBTAARWAAI7AAQ3AABKAAM+AIEmsEAASpBTOwA3LwFHRABPSAI
      7KgYyIzMyAARPAAM3AAE7AABHAANTACxHOABPRAJTLwU3NgQ7MwIyLCNTAARPAAAyAAJHAAQ3AA
      Y7AIEANzADOy8BT0oDR0AEU0ECMh4cNwAHMgABUwAATwADOwACRwAeTi4GRTcAPDQDOTgBMiMAU
      SZYOQAEUQADRQABMgACTgADPABdTkADUTsEOTUAPCsBMjABRTktMgANOQAERQANUQAAPAAGTgBz
      TE0DPDIBSD4BNzECT0IAQCoCQSMzQQAHQAABPAADNwAPSAAETAAiTEMCSC8CPDgEQCwDNywmSAA
      ATAAGNwAHPAACQAAETwCBBzxAAEAwAkhABE9AAExJAzclGTcABEAACUgABDwAD0wACkcxBEozBT
      s+BD4tATcngVKwQH9WkEoAAUcACTcAED4ABDsAJk8AJk8wBEo3A0c0ADs6Aj4wArBAAACQNzKBF
      bBAf4EQkEcABkoAGk8AFzsADz4AHjcAgUlDKwBTTQNPUAA+KQJKPwA7LQU3IwmwQABNkD4ABjcA
      A08ABkoAB0MACjsAVjw4AzczAUAwAlREAU9GA0g6FlMAgR2wQH8tkE8ABkgAIDcAA0AAIDwAF1M
      /Ak9IAUo3A7BAAAKQOy8HPiwENyoGVABpsEB/dJBTAAZPABNKAA07ABA+AAM3AIEtsEAAFJA3Mw
      BPSANHPgBKPwE7LQEyLYFXsEB/B5BKAAI3AAdHAAwyAAg7ADBPAECwQAAMkE4nC0gyAjwlADYUA
      UUsAjkqCjIagSIyAARFAAc2AAI5AAJIAAVOAAywQH8NkDwAfrBAADmQPEQCQDQBNzwBSFAATFkD
      T1JCNwAEQAAHPAACSAAETwAATAAiTD8ETy0APDMCSCsBQCsENysgSAAATAABTwAFNwADPAAEQAA
      MTzQDTEcBSC8JPDYAQCsBNy1SNwAATwADQAAESAATPAATSj8ATzoDR0IBTAADOzICNzEEPi9psE
      B/U5BHAAZKABFPAB47AA83AAY+AIFDsEAAGZBKOwI+RQA3RgBTNwJPNgA7RAFROUxRAGxKAAFPA
      AVTAAQ3AAA7AAM+AIRqTk4DU1UDSkkAT1ABOzwCPjIDNzUqTgCBGUoAAU8AADcAA1MABjsABz4A
      hEBOTAJTUQJPUgBKRQg7QgM+OQE3PhBOAIEMOwAANwADSgAGTwAEPgAHUwCBT05QBFNTAEpJAk9
      WAj4yAjcxADs1HE4AgRo7AARPAAA3AAZKAAM+AARTAIFPTlQEU1UASk0CT1gCOywANyoCPi4jTg
      BATwATUwAESgATNwADOwAGPgCFEU5UAkpNAU9cAVNZADs6ATc1BD4vGU4AF0oAEjcAATsAAz4AC
      VMAAU8AhUxPVgFKVQBJUgNHUgI7QgE3OAM+MRpJAFxKAARHAAxPAAM7AAg+AAA3AIUeTlAGSkEA
      U1UBT1IEPjICO0YBNz4dTgBPSgAKOwAANwAHTwAMPgAHUwCBdk9KA1JPAFZSAlNTAD4zATcsATs
      2HFIAcE8ABlMABFYAAzsAAzcACT4AgWROSgM7MQE3KgBTVQFKRQJPUAE+Kx9OAFVPAARKAANTAA
      o7AAM3AAo+AIF2T1QASU4BR0oCOzQANzADPjAASk8jSQBKRwAGSgATNwADOwABTwAMPgCBfVNTA
      E5OBEpJAE9WAztEAjc2AT4yGU4Ad0oAA08ABlMABDsABj4AADcAhSZPTgJJPgBHQgFKTQQ7LQA+
      LgI3LBpJAGQ7AAI3AAJKAAVHAAY+AARPAIR2TlAEU1MDT1QASkkGPjkBOzgCNzEXTgBNNwAQTwA
      COwAHUwADSgADPgCCBE9OAkpJAj4tAklKAEdKAzstBDcfIkkAfkcABjcABkoAAzsABD4ACU8AgV
      dOTAZTUwBKOQBPUgNMRQQ+LAI7MAE3KhlOABNMAGA3AANPAANKAANTAAE7AAk+AIFWSUwER0YDS
      k8BT04CPjMDNzEBOzQZSQBgSgAWRwAHNwADOwADTwAHPgCBbUpDAkdIBEM7ADc6ATtCAj5BgT2w
      QH9DkEMAIEcAEEoALzcACzsABj4AgW9PVgRHUABKVQdDMQM7MgA+LQM3Lzc+AABHAAM3AANKAAN
      DAA07AHmwQAACkExLADw8AUNJAEhKAUBCBDc1FU8AgUOwQH8UkEgAAjcACkAAA0MATTwAMEwAA7
      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
      SlsDO0QAQ2EDPlMDN0xtQwADPgAGSgAANwABRwAEOwCBeUdeAjtCAkNjAT5TAjdOgQs3AAFDAAI
      +AANHAAE7AIIQR14AO0QCPk8AQ2EBN0p6QwAFPgABRwAFNwAKOwCBckNZAUdUBDdIADtAAT5JPT
      4AAUMAATcAAzsAA0cAfUdKB0NTAj5DAzc6ATsvLUMABT4ABEcABjsAATcAgQNHVgRDWwBKUwI+S
      QM7PgE3Rl+xQH9ykT4ABkMACUcAEkoABDcAHDsAgmA7OgBPXgBKWQA+NwNDVQE3M0BDAA07AAY3
      AARPAAVKAAU+AGBPXAVMXwJDWQE8QgRAPAE3QAixQACBM0B/QpFDAAQ3AAVPAAFMAB1AABc8AE9
      PWANKVQQ7MwA+MgE3LwJDUQexQACBJkB/DJFPAANKAAc+AAE7AAVDAAI3AIFLSkECR1IGNz4APk
      EBOz4OsUAAgTeRNwAJsUB/HpE7AChKAAY+ABhHAFBITgRFTwI+PwQ8MQE2HQE5MwUyHw6xQACBP
      pFIAAE+AAYyAAVFAAE2AACxQH8JkTwAAzkAgStHSgGxQAACkUNNBT45AzcjAzsshCU+AANDAAxH
      ABA3ABk7AIE+T0gISk8CQ00BOzMCPjMCNy2BRrFAf1SRQwALSgASTwAfOwAdNwACPgCBV1ZSBU9
      UAkpHBUM7BTs0BjcwNlYAAUMAAk8ABEoAATcAAjsAarFAAAeRVF4ET14BSFQBPC8CQDIENy2BVr
      FAfyqRNwAASAAKTwArQAAUPAA9VAAHU0kFSjoAT0wEPi8ENysAOzAPsUAAgSuRUwADTwAMsUB/B
      pFKAB07AAM3ABY+AIEPT1IISlECR0wCPjAGOzQDsUAAgVdAfwyRRwADSgADTwABOwASPgCBCU46
      A0o9BzYMATwhAEUwCbFAAAaRPh6BOUoACDwAAU4ABUUAADYACbFAfweRPgCBA7FAACWRTFEFSEw
      DPDAGQCkANyoqSAAMNwABTAACQAABPAB4TEEFSEIDPDMFQDACNys1NwAFSAABQAADTAAXPABmSj
      QERzUHOyoFPisENyeBMLFAfxqRRwACSgAeNwAGOwAVPgCBBE9AA0pDArFAAASROzAAQzgAPjECN
      y2CPLFAf26RNwAjTwAFSgAIQwAePgANOwBST1YDSlUEQ1ECO0QAPjRjQwACSgACTwASPgAAOwBZ
      TFcCT1QCPCoCQ08BQDECNykbsUAAgQ9Af1yRQwAKTwAKTAABNwAOQAAOPABJSjgHTzUEPjIBQ0M
      CNywDOy8ksUAAT0B/K5FDAA5KABNPAAE3AAY7AAk+AIFPU1cBT1YBSkkFOzQCNzgQsUAAgQ1Af1
      CRTwAINwACUwAEOwAHSgCBAk4vBVE1BEg0BTwpAzceHbFAAIEDQH8TkTcAB1EAAk4ADkgADTwAg
      Q2xQAAjkUhIAU9GAkAnADwpAUU0BjcfJjcAAEUAAUgAA08AAjwACUAAgQNPOwJFOwFITgdAMAA8
      NAc3KylIAAJFAAtAAAc3AAk8AG5HOgRDPwE+LgQ7LAI3J3qxQH9XkTsAAzcAEj4ABEcAFU8ADEM
      AVbFAACCRSlMFR1oFPjYAQ0kBO0ICNzZ4QwADRwAFSgAHOwAENwAFPgCFJ0pXBEdaAUNPAj4yAT
      s1ATc0a0MAA0cABUoABDsAAzcAAD4AhSxKVwZDUwA+NQFHXAE7QgI3OG1HAARDAAVKAAE7AAA3A
      Ak+AIIMSlUBPjcAO0ICR1oCNzYAQ1VzQwAARwAFSgABNwAEOwAIPgCCDUpTAUdaA0NTBzs8AT4z
      CDc4cEcAAEoAAkMABzcAAzsAAz4AhR1KVwNHXANDVQI7QAI+NAE3PGVHAAJDAAc7AABKAAI3AAE
      +AIU8T14DSlkDPjsAQ1kBO0YFN0BiSgADTwACQwAFOwAANwAIPgCFNk9eCEpdAj4xAUNdAjs0AT
      c8PkMAEkoAAE8AATsAAjcABj4AghlPXAJKVwRDVwU7RAM3NgM+L1JDAAI3AAJKAAM+AABPAAM7A
      IImT1oDSlkCQ1cHOz4CPjYBNzVbSgADQwAETwAENwADOwACPgCFOkpHAk9GBENDAT4zAzsxATct
      gRyxQH97kUMACEoAEk8AHjsADTcABT4AgXlTWwZKQwNDOAI+MQE7MAQ3Kjg3AAE+AAhDAAI7AAZ
      KAHNUYAVMSQdANwE8NQM3LwGxQAADkVMAgRyxQH8zkTcAH0AACkwADTwAfVNNBUoxBD4rCDssAz
      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
      Td3d3LndpemFyZC5uZXQvfmRzcgD/LwA=`;
  }

  initialState() {
    return {
      selected_sheet: "",
      uploader_container_className: "hidden",
      music_playing: false
    };
  }

  onSelectedSheetChange(event) {
    event.preventDefault();
    let sheet = event.target.value;
    this.setState({ selected_sheet: sheet });
    if (sheet === "") {
      this.resetOsmd();
      return;
    } else {
      axios.get(settings.endpoints.get_music_xml,
        {
          params: {
            sheet_name: sheet
          }
        }
      )
        .then(response => {
          let xml = response.data;
          if (xml !== "") {
            this.renderOsmd(xml);
          }
        });
    }
  }

  renderOsmd(xml) {
    this.resetOsmd();
    console.log('rendering xml');
    let osmd = new window.opensheetmusicdisplay.OpenSheetMusicDisplay("osmd", true, "svg");
    osmd.load(xml).then(
      function () {
        osmd.render();
      }
    );
  }

  toggleUploader() {
    let bool = (this.state.uploader_container_className === "open") ? "hidden" : "open";
    this.setState({ uploader_container_className: bool })
  }

  togglePlay() {
    let playing = (this.state.music_playing === true) ? false : true;
    if (this.state.music_playing === false) { // will be set to true now
      this.playMidi();
    } else {
      this.stopMidi();
    }
    this.setState({ music_playing: playing });
  }

  renderUploader() {
    let ctx = this;
    let to_wait = 2000;
    window.Dropzone.options.uploader = {
      url: settings.endpoints.image_upload,
      init: function () {
        this.on("success", function (file, response) {
          console.log('response has arrived from server');
          let parsed = JSON.parse(response).data;
          let xml = parsed.xml;

          console.log(xml);
          let cleaned = xml.replace(/<part-name \/>/ig, '');
          let cleaned2 = cleaned.replace(/<score-part id=(['"\\A-Za-z0-9]+)>/ig, `<score-part id=$1><part-name print-object="no">Piano</part-name>`);
          // let cleaned = xml;
          console.log(cleaned2);

          setTimeout(function () {
            ctx.toggleUploader();
            ctx.renderOsmd(cleaned2);

            ctx.midi = parsed.midi;
            ctx.playMidi();
            ctx.setState({ music_playing: true });

          }, to_wait);

        });
        // this.on("complete", function (file) {
        //   setTimeout(function () {
        //   this.removeFile(file);
        //   }, to_wait+200);
        // });
      },
      // clickable: window.document.getElementsByClassName("dz-default dz-message")[0],
      clickable: false,
      // maxFiles: 1,
      thumbnailWidth: 250,
      thumbnailHeight: 250,
      dictDefaultMessage: "<b>Drop images here</b> to upload them 😂",
      timeout: 999999
    };
  }

  playMidi() {
    console.log('playing midi');
    let $ = window.$;
    let el = $("#midiPlayer");
    el.midiPlayer({
      // onUpdate: midiUpdate,
      // onStop: midiStop,
      // width: 250
    });
    // var song = 'data:audio/midi;base64,TVRoZAAAAAYAAQABBABNVHJrAAACNAD/AwAA4ABAAJBNWqAAgE0AAJBFWoQAgEUAAJBHWogAgEcAAJBAWogAgEAAAJBAWoQAgEAAAJA8WoQAgDwAhACQTVqEAIBNAACQSlqQAIBKAACQTVqEAIBNAACQTVqIAIBNAACQTVqIAIBNAACQT1qIAIBPAACQSlqQAIBKAACQR1qQAIBHAACQR1qEAIBHAACQSlqIAIBKAACQTVqEAIBNAACQT1qQAIBPAACQT1qEAIBPAACQSlqIAIBKAACQR1qEAIBHAACQRVqIAIBFAACQPFqQAIA8AACQQVqEAIBBAACQQFqEAIBAAACQRVqQAIBFAACQSlqEAIBKAACQRVqIAIBFAACQQ1qEAIBDAACQR1qQAIBHAACQSlqEAIBKAACQQVqIAIBBAACQQ1qEAIBDAACQRVqQAIBFAACQRVqEAIBFAACQR1qIAIBHAACQQFqEAIBAAACQTVqIAIBNAACQRVqEAIBFAACQRVqEAIBFAACQQ1qEAIBDAACQQ1qIAIBDAACQPFqEAIA8AACQT1qgAIBPAACQQVqIAIBBAACQPFqIAIA8AACQQFqQAIBAAACQSlqgAIBKAACQPlqgAIA+AACQR1qgAIBHAACQQFqgAIBAAACQU1qgAIBTAACQQFqEAIBAAACQPlqEAIA+AACQR1qIAIBHAACQQVqIAIBBAACQSlqEAIBKAIQAkE9akACATwAAkExahACATAAAkExaiACATAAAkE1ahACATQCIAP8vAA==';
    // el.midiPlayer.play(song);
    el.midiPlayer.play(this.midiPrefix + this.midi);
  }

  stopMidi() {
    let $ = window.$;
    let el = $("#midiPlayer");
    // var song = 'data:audio/midi;base64,TVRoZAAAAAYAAQABBABNVHJrAAACNAD/AwAA4ABAAJBNWqAAgE0AAJBFWoQAgEUAAJBHWogAgEcAAJBAWogAgEAAAJBAWoQAgEAAAJA8WoQAgDwAhACQTVqEAIBNAACQSlqQAIBKAACQTVqEAIBNAACQTVqIAIBNAACQTVqIAIBNAACQT1qIAIBPAACQSlqQAIBKAACQR1qQAIBHAACQR1qEAIBHAACQSlqIAIBKAACQTVqEAIBNAACQT1qQAIBPAACQT1qEAIBPAACQSlqIAIBKAACQR1qEAIBHAACQRVqIAIBFAACQPFqQAIA8AACQQVqEAIBBAACQQFqEAIBAAACQRVqQAIBFAACQSlqEAIBKAACQRVqIAIBFAACQQ1qEAIBDAACQR1qQAIBHAACQSlqEAIBKAACQQVqIAIBBAACQQ1qEAIBDAACQRVqQAIBFAACQRVqEAIBFAACQR1qIAIBHAACQQFqEAIBAAACQTVqIAIBNAACQRVqEAIBFAACQRVqEAIBFAACQQ1qEAIBDAACQQ1qIAIBDAACQPFqEAIA8AACQT1qgAIBPAACQQVqIAIBBAACQPFqIAIA8AACQQFqQAIBAAACQSlqgAIBKAACQPlqgAIA+AACQR1qgAIBHAACQQFqgAIBAAACQU1qgAIBTAACQQFqEAIBAAACQPlqEAIA+AACQR1qIAIBHAACQQVqIAIBBAACQSlqEAIBKAIQAkE9akACATwAAkExahACATAAAkExaiACATAAAkE1ahACATQCIAP8vAA==';
    // el.midiPlayer.stop(song);
    el.midiPlayer.stop(this.midiPrefix + this.midi);
  }

  resetOsmd() {
    let q = window.document.querySelector('div[id="osmd"]');
    while (q.hasChildNodes()) {
      q.removeChild(q.lastChild);
    }
    window.document.querySelector('div[id="osmd"]').innerHTML = "";
  }

  componentDidMount() {
    this.renderUploader();
    axios.get(settings.server_url);
  }

  render() {
    return (
      <div className="App">

        <div className="header">
          <div className="left">
            <img src={logo} className="logo" alt="logo" />
            <div className="title">
              Musically
            </div>
            <div className="round-btn" onClick={this.toggleUploader}>
              <div className="round-btn-inside">
                <img src={upload_icon} className="upload-svg" alt="Upload files" />
              </div>
            </div>
            <div className="play-btn round-btn" onClick={this.togglePlay}>
              <div className="round-btn-inside">
                <img src={this.state.music_playing === true ? pause_icon : play_icon} className="play-svg" alt="Play the music" />
              </div>
            </div>
          </div>

          <div className="right">
            <div className="credits">
              <span>By Angus, Annie and Yiwei</span>
            </div>
          </div>
        </div>

        <div className={"dropzone-container " + this.state.uploader_container_className} onClick={this.toggleUploader}>
          <div id="uploader" className="dropzone"></div>
        </div>

        <div id="midiPlayer" style={{ display: 'none' }}></div>

        <br />

        <div id="osmd">
          {/* <div id="sample" style={{
            height: '400px',
            width: '220px',
            display: 'flex',
            'flexDirection': 'column',
            'textAlign': 'center',
            'justifyContent': 'center',
            margin: 'auto' 
          }}>
          <div style={{ fontSize: 14 + 'pt', 'padding': '10px' }}>Select a sample sheet to get started!</div>
          <select name="selected-sheet"
            onChange={this.onSelectedSheetChange}
            value={this.state.selected_sheet}
            style={{ margin: '0 auto' }}>
            <option value=""></option>
            <option value="test">test</option>
            <option value="ActorPreludeSample">Chant</option>
            <option value="BeetAnGeSample">BeetAnGeSample</option>
            <option value="HelloWorld">HelloWorld</option>
            <option value="MozartPianoSonata">MozartPianoSonata</option>
          </select>
          </div> */}
        </div>
      </div>
    );
  }
}

export default App;
