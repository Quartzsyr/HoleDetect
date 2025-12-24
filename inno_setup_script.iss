; 脚本由 Inno Setup 脚本向导生成
; 完善版 - 添加开发者信息显示
; 有关创建 Inno Setup 脚本文件的详细信息请查阅帮助文档

#define MyAppName "【最终版】孔洞检测程序"
#define MyAppVersion "2.0.1"
#define MyAppPublisher "石殷睿"
#define MyAppURL "https://www.quartz.xin"
#define MyAppExeName "孔洞检测程序.exe"
#define MyDeveloper "苏州大学 | 石殷睿"
#define MyProject "2025光电设计大赛作品"

[Setup]
; 注意: AppId的值为单独标识这个应用程序
; 不要为其他安装程序使用相同的AppId值
AppId={{A5E31B2D-1C5F-4A25-B7C5-2E4F58D1C92A}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
; 以下行取消注释，以在非管理安装模式下运行（仅为当前用户安装）
;PrivilegesRequired=lowest
OutputBaseFilename=【最终版】孔洞测量安装包{#MyAppVersion}
OutputDir=.
SetupIconFile=icon.ico
SolidCompression=yes
WizardStyle=modern
Compression=lzma/ultra
InternalCompressLevel=ultra

; 添加版权信息
AppCopyright=Copyright © 2025 {#MyAppPublisher}
; 添加版本信息
VersionInfoVersion={#MyAppVersion}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription={#MyAppName}
VersionInfoCopyright=Copyright © 2025 {#MyAppPublisher}
; 自定义安装向导窗口大小
WizardSizePercent=120
; 显示欢迎页面和完成页面
DisableWelcomePage=no
DisableFinishedPage=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "创建快速启动图标"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "*.iss"
; 注意: 不要在任何共享系统文件上使用"Flags: ignoreversion"

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Comment: "{#MyProject}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; Comment: "{#MyProject}"
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[CustomMessages]
; 中文消息（使用英文语言但显示中文内容）
english.WelcomeLabel2=欢迎使用 [name] 安装向导！%n%n这是由 {#MyDeveloper} 开发的 {#MyProject}。%n%n本程序专为光电检测应用设计，具有高精度的孔洞识别能力。%n%n建议您在继续安装前关闭所有其他应用程序。
english.FinishedLabelNoIcons=安装程序已在您的计算机上成功安装 [name]。%n%n开发者：{#MyDeveloper}%n项目：{#MyProject}%n版本：{#MyAppVersion}%n%n感谢您的使用！
english.FinishedLabel=安装程序已在您的计算机上成功安装 [name]。%n%n开发者：{#MyDeveloper}%n项目：{#MyProject}%n版本：{#MyAppVersion}%n%n您可以通过选择安装的图标来运行此程序。%n%n感谢您的使用！
english.AboutApp=关于 %1
english.DeveloperInfo=开发者：{#MyDeveloper}
english.DevelopedFor={#MyProject}

[Code]
// 在安装开始前显示开发者信息
function InitializeSetup(): Boolean;
begin
  Result := True;
  if MsgBox('孔洞检测程序 v{#MyAppVersion}' + #13#10#13#10 +
            '开发者：{#MyDeveloper}' + #13#10 +
            '项目：{#MyProject}' + #13#10#13#10 +
            '本程序为光电设计大赛设计:D' + #13#10#13#10 +
            '是否继续安装？', 
            mbInformation, MB_YESNO) = IDNO then
    Result := False;
end;

// 自定义安装进度页面显示
procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpInstalling then
  begin
    WizardForm.StatusLabel.Caption := '正在安装' + ' {#MyAppName}...';
    WizardForm.FilenameLabel.Caption := '开发者：{#MyDeveloper}';
  end;
end;

// 在卸载时显示开发者信息
function InitializeUninstall(): Boolean;
begin
  Result := True;
  if MsgBox('即将卸载 {#MyAppName}' + #13#10#13#10 +
            '开发者：{#MyDeveloper}' + #13#10 +
            '看来这版又不行/(ㄒoㄒ)/~~' + #13#10#13#10 +
            '确定要卸载吗？', 
            mbConfirmation, MB_YESNO) = IDNO then
    Result := False;
end;