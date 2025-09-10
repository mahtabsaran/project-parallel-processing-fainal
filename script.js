// داده‌های ابزارهای همزمانی
const threadTools = [
    { id: 1, name: "Defining a thread" },
    { id: 2, name: "Determining the current thread" },
    { id: 3, name: "Thread subclass" },
    { id: 4, name: "Synchronization with Lock" },
    { id: 5, name: "Synchronization with RLock" },
    { id: 6, name: "Synchronization with Semaphores" },
    { id: 7, name: "Synchronization with Barrier" }
];

const processTools = [
    { id: 1, name: "Spawning a process" },
    { id: 2, name: "Naming a process" },
    { id: 3, name: "Running processes in background" },
    { id: 4, name: "Killing a process" },
    { id: 5, name: "Process subclass" },
    { id: 6, name: "Using queue for data exchange" },
    { id: 7, name: "Synchronizing processes" },
    { id: 8, name: "Using process pool" }
];

// المنت‌های DOM
const methodSelect = document.getElementById('methodSelect');
const toolSelect = document.getElementById('toolSelect');
const scenarioSelect = document.getElementById('scenarioSelect');
const scenarioForm = document.getElementById('scenarioForm');
const resultsDiv = document.getElementById('results');
const loadingDiv = document.getElementById('loading');

// آدرس API - با توجه به محیط اجرا تغییر دهید
const API_BASE_URL = 'https://parallel-processing-backend.onrender.com'; // یا آدرس سرور شما

// رویداد تغییر نوع پردازش
methodSelect.addEventListener('change', function() {
    const method = this.value;
    toolSelect.disabled = !method;
    scenarioSelect.disabled = true;
    scenarioSelect.innerHTML = '<option value="">-- ابتدا ابزار را انتخاب کنید --</option>';

    if (method) {
        // پر کردن ابزارهای مربوطه
        const tools = method === 'thread' ? threadTools : processTools;
        toolSelect.innerHTML = '<option value="">-- انتخاب کنید --</option>';

        tools.forEach(tool => {
            const option = document.createElement('option');
            option.value = tool.id;
            option.textContent = tool.name;
            toolSelect.appendChild(option);
        });
    } else {
        toolSelect.innerHTML = '<option value="">-- ابتدا نوع پردازش را انتخاب کنید --</option>';
    }
});

// رویداد تغییر ابزار
toolSelect.addEventListener('change', function() {
    const toolId = this.value;
    scenarioSelect.disabled = !toolId;

    if (toolId) {
        // پر کردن سناریوها (همیشه 3 سناریو)
        scenarioSelect.innerHTML = '<option value="">-- انتخاب کنید --</option>';
        for (let i = 1; i <= 3; i++) {
            const option = document.createElement('option');
            option.value = i;
            option.textContent = `سناریو ${i}`;
            scenarioSelect.appendChild(option);
        }
    } else {
        scenarioSelect.innerHTML = '<option value="">-- ابتدا ابزار را انتخاب کنید --</option>';
    }
});

// ارسال فرم
scenarioForm.addEventListener('submit', async function(e) {
    e.preventDefault();

    const method = methodSelect.value;
    const toolId = toolSelect.value;
    const scenario = scenarioSelect.value;

    if (!method || !toolId || !scenario) {
        showToast('لطفاً تمام فیلدها را پر کنید', 'error');
        return;
    }

    // نمایش لودینگ
    loadingDiv.style.display = 'block';
    resultsDiv.style.display = 'none';

    try {
        // پیدا کردن نام ابزار
        const toolName = method === 'thread'
            ? threadTools.find(t => t.id == toolId).name
            : processTools.find(t => t.id == toolId).name;

        // ارسال درخواست به سرور
        const response = await fetch(`${API_BASE_URL}/api/run-scenario`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                section_type: method,
                section_name: toolName,
                scenario_number: parseInt(scenario),
                parameters: {}
            })
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `خطای سرور: ${response.status}`);
        }

        const data = await response.json();

        // نمایش نتایج
        showResults(data, method, toolId, scenario, toolName);

    } catch (error) {
        console.error('Error:', error);
        showToast(`خطا در ارتباط با سرور: ${error.message}`, 'error');
    } finally {
        // مخفی کردن لودینگ
        loadingDiv.style.display = 'none';
    }
});

// نمایش نتایج
function showResults(data, method, toolId, scenario, toolName) {
    // پر کردن اطلاعات نتایج
    document.getElementById('resultMethod').textContent =
        method === 'thread' ? 'Thread-Based Parallelism' : 'Process-Based Parallelism';

    document.getElementById('resultTool').textContent = toolName;
    document.getElementById('resultScenario').textContent = scenario;
    document.getElementById('resultTime').textContent = new Date().toLocaleTimeString('fa-IR');

    // نمایش خروجی
    const outputContent = document.getElementById('outputContent');
    if (data.output && Array.isArray(data.output)) {
        outputContent.textContent = data.output.join('\n');
    } else {
        outputContent.textContent = 'خروجی نامعتبر دریافت شد';
    }

    // نمایش توضیحات
    const explanationElement = document.getElementById('explanation');
    explanationElement.textContent = data.explanation || 'توضیحاتی ارائه نشده است';

    // نمایش بخش نتایج
    resultsDiv.style.display = 'block';

    // اسکرول به نتایج
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

// تابع نمایش toast notification
function showToast(message, type = 'info') {
    // ایجاد یک toast ساده
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        z-index: 1000;
        min-width: 300px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    `;

    // رنگ‌های مختلف برای انواع پیام
    const colors = {
        success: '#28a745',
        error: '#dc3545',
        warning: '#ffc107',
        info: '#17a2b8'
    };

    toast.style.backgroundColor = colors[type] || colors.info;
    toast.textContent = message;

    // اضافه کردن toast به صفحه
    document.body.appendChild(toast);

    // حذف خودکار بعد از 5 ثانیه
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 5000);
}

// تابع برای تست ارتباط با سرور
async function testConnection() {
    try {
        const response = await fetch(`${API_BASE_URL}/`);
        if (response.ok) {
            console.log('اتصال با سرور برقرار است');
            return true;
        }
    } catch (error) {
        console.warn('اتصال با سرور برقرار نیست:', error.message);
        return false;
    }
}

// تست ارتباط هنگام لود صفحه
document.addEventListener('DOMContentLoaded', function() {
    testConnection().then(isConnected => {
        if (!isConnected) {
            showToast('سرور در دسترس نیست. لطفاً مطمئن شوید FastAPI در حال اجراست.', 'warning');
        }
    });
});

// تابع کمکی برای فرمت‌دهی خروجی
function formatOutput(outputArray) {
    if (!Array.isArray(outputArray)) return '';
    return outputArray.join('\n');
}